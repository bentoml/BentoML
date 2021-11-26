from typing import Callable, Iterable, Optional, Tuple, Union
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from .trainer_utils import SchedulerType

logger = ...

def get_constant_schedule(optimizer: Optimizer, last_epoch: int = ...): ...
def get_constant_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = ...
): ...
def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=...
): ...
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = ...,
    last_epoch: int = ...,
): ...
def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = ...,
    last_epoch: int = ...,
): ...
def get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    lr_end=...,
    power=...,
    last_epoch=...,
): ...

TYPE_TO_SCHEDULER_FUNCTION = ...

def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = ...,
    num_training_steps: Optional[int] = ...,
): ...

class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = ...,
        betas: Tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
        correct_bias: bool = ...,
    ) -> None: ...
    def step(self, closure: Callable = ...): ...

class Adafactor(Optimizer):
    def __init__(
        self,
        params,
        lr=...,
        eps=...,
        clip_threshold=...,
        decay_rate=...,
        beta1=...,
        weight_decay=...,
        scale_parameter=...,
        relative_step=...,
        warmup_init=...,
    ) -> None: ...
    def step(self, closure=...): ...

class AdafactorSchedule(LambdaLR):
    def __init__(self, optimizer, initial_lr=...) -> None: ...
    def get_lr(self): ...

def get_adafactor_schedule(optimizer, initial_lr=...): ...
