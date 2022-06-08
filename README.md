# hf-sharpness
Simple implementation of flat minima methods (SAM, fisher penalty) for Huggingface trainer.

Replace your `Trainer` class with `BaseTrainer`, and use our `TrainingArguments`:

```
from nlpsharpness import BaseTrainer, TrainingArguments
training_args = TrainingArguments(
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    gradient_accumulation_steps=2,
    sam=True,                        # Use sharpness aware minimization
    sam_rho=0.01,                    # Step size for SAM
    fisher_penalty_weight=0.01,      # Use Fisher penalty with this weight
)
```
The `evaluate_hessian(dataset)` method of `BaseTrainer` returns the largest eigenvalue and trace of the Hessian for `dataset`.
