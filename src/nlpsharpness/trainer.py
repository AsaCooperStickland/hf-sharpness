from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import time
import torch
import collections
from packaging import version
from torch.distributions import Categorical
import torch.nn as nn

from transformers import Trainer
from transformers import logging
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import (
    get_parameter_names,
)
from nlpsharpness.sharpness import SAM


if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)


class BaseTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=self.optimizer
        )

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in decay_parameters
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in decay_parameters
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            """if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:"""
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
        if self.args.sam:
            self.optimizer = SAM(
                optimizer_grouped_parameters,
                optimizer_cls,
                rho=float(self.args.sam_rho),
                adaptive=self.args.sam_adaptive,
                **optimizer_kwargs
            )
        # if is_sagemaker_mp_enabled():
        #    self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        """if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)"""

        def step():
            with self.autocast_smart_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()
            return loss

        loss = step()

        if self.args.sam:
            self.optimizer.first_step(zero_grad=True)

            loss = step()
            self.optimizer.assign_grads()
            self.optimizer.zero_grad()
        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if isinstance(self.args.fisher_penalty_weight, str):
            self.args.fisher_penalty_weight = float(self.args.fisher_penalty_weight)
        if self.args.fisher_penalty_weight > 0.0 and model.training():
            logits = outputs.get("logits")
            outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
            f_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            f_loss = f_loss_fct(logits.view(-1, logits.size(-1)), outdx.view(-1))
            # f_loss_fct = nn.CrossEntropyLoss()
            # f_loss = f_loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            grads = torch.autograd.grad(
                f_loss,
                [p for n, p in model.named_parameters() if p.requires_grad == True],
                retain_graph=True,
                create_graph=True,
            )
            gr_norm_sq = 0.0
            for gr in grads:
                if gr is not None:
                    gr_norm_sq += (gr**2).sum()
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if self.args.fisher_penalty_weight > 0.0 and model.training():
            loss += (
                self.args.fisher_penalty_weight
                * gr_norm_sq
                / self.args.train_batch_size
            )

        return (loss, outputs) if return_outputs else loss
