from typing import Callable, List, Optional, Union
import tensorflow as tf

class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        power: float = ...,
        name: str = ...,
    ) -> None: ...
    def __call__(self, step): ...
    def get_config(self): ...

def create_optimizer(
    init_lr: float,
    num_train_steps: int,
    num_warmup_steps: int,
    min_lr_ratio: float = ...,
    adam_beta1: float = ...,
    adam_beta2: float = ...,
    adam_epsilon: float = ...,
    weight_decay_rate: float = ...,
    power: float = ...,
    include_in_weight_decay: Optional[List[str]] = ...,
): ...

class AdamWeightDecay(tf.keras.optimizers.Adam):
    def __init__(
        self,
        learning_rate: Union[
            float, tf.keras.optimizers.schedules.LearningRateSchedule
        ] = ...,
        beta_1: float = ...,
        beta_2: float = ...,
        epsilon: float = ...,
        amsgrad: bool = ...,
        weight_decay_rate: float = ...,
        include_in_weight_decay: Optional[List[str]] = ...,
        exclude_from_weight_decay: Optional[List[str]] = ...,
        name: str = ...,
        **kwargs
    ) -> None: ...
    @classmethod
    def from_config(cls, config): ...
    def apply_gradients(self, grads_and_vars, name=..., **kwargs): ...
    def get_config(self): ...

class GradientAccumulator:
    def __init__(self) -> None: ...
    @property
    def step(self): ...
    @property
    def gradients(self): ...
    def __call__(self, gradients): ...
