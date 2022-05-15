from transformers import TrainingArguments
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class TrainingArguments(TrainingArguments):
    sam: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use sharpness aware minimization"}
    )
    sam_rho: Optional[float] = field(
        default=0.05, metadata={"help": "Step size for SAM"}
    )
    sam_adaptive: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use adaptive SAM"}
    )
    fisher_penalty_weight: Optional[float] = field(
        default=0.001, metadata={"help": "Weight for fisher penalty loss term"}
    )