

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from .training_args import TrainingArguments

logger = ...
@dataclass
class TrainerState:
    """
    A class containing the :class:`~transformers.Trainer` inner state that will be saved along the model and optimizer
    when checkpointing and passed to the :class:`~transformers.TrainerCallback`.

    .. note::

        In all this class, one step is to be understood as one update step. When using gradient accumulation, one
        update step may require several forward and backward passes: if you use :obj:`gradient_accumulation_steps=n`,
        then one update step requires going through `n` batches.

    Args:
        epoch (:obj:`float`, `optional`):
            Only set during training, will represent the epoch the training is at (the decimal part being the
            percentage of the current epoch completed).
        global_step (:obj:`int`, `optional`, defaults to 0):
            During training, represents the number of update steps completed.
        max_steps (:obj:`int`, `optional`, defaults to 0):
            The number of update steps to do during the current training.
        total_flos (:obj:`float`, `optional`, defaults to 0):
            The total number of floating operations done by the model since the beginning of training (stored as floats
            to avoid overflow).
        log_history (:obj:`List[Dict[str, float]]`, `optional`):
            The list of logs done since the beginning of training.
        best_metric (:obj:`float`, `optional`):
            When tracking the best model, the value of the best metric encountered so far.
        best_model_checkpoint (:obj:`str`, `optional`):
            When tracking the best model, the value of the name of the checkpoint for the best model encountered so
            far.
        is_local_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
            several machines) main process.
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not this process is the global main process (when training in a distributed fashion on several
            machines, this is only going to be :obj:`True` for one process).
        is_hyper_param_search (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether we are in the process of a hyper parameter search using Trainer.hyperparameter_search. This will
            impact the way data will be logged in TensorBoard.
    """
    epoch: Optional[float] = ...
    global_step: int = ...
    max_steps: int = ...
    num_train_epochs: int = ...
    total_flos: float = ...
    log_history: List[Dict[str, float]] = ...
    best_metric: Optional[float] = ...
    best_model_checkpoint: Optional[str] = ...
    is_local_process_zero: bool = ...
    is_world_process_zero: bool = ...
    is_hyper_param_search: bool = ...
    trial_name: str = ...
    trial_params: Dict[str, Union[str, float, int, bool]] = ...
    def __post_init__(self):
        ...
    
    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside :obj:`json_path`."""
        ...
    
    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of :obj:`json_path`."""
        ...
    


@dataclass
class TrainerControl:
    """
    A class that handles the :class:`~transformers.Trainer` control flow. This class is used by the
    :class:`~transformers.TrainerCallback` to activate some switches in the training loop.

    Args:
        should_training_stop (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the training should be interrupted.

            If :obj:`True`, this variable will not be set back to :obj:`False`. The training will just stop.
        should_epoch_stop (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the current epoch should be interrupted.

            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next epoch.
        should_save (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should be saved at this step.

            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next step.
        should_evaluate (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should be evaluated at this step.

            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next step.
        should_log (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the logs should be reported at this step.

            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next step.
    """
    should_training_stop: bool = ...
    should_epoch_stop: bool = ...
    should_save: bool = ...
    should_evaluate: bool = ...
    should_log: bool = ...


class TrainerCallback:
    """
    A class for objects that will inspect the state of the training loop at some events and take some decisions. At
    each of those events the following arguments are available:

    Args:
        args (:class:`~transformers.TrainingArguments`):
            The training arguments used to instantiate the :class:`~transformers.Trainer`.
        state (:class:`~transformers.TrainerState`):
            The current state of the :class:`~transformers.Trainer`.
        control (:class:`~transformers.TrainerControl`):
            The object that is returned to the :class:`~transformers.Trainer` and can be used to make some decisions.
        model (:class:`~transformers.PreTrainedModel` or :obj:`torch.nn.Module`):
            The model being trained.
        tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer used for encoding the data.
        optimizer (:obj:`torch.optim.Optimizer`):
            The optimizer used for the training steps.
        lr_scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`):
            The scheduler used for setting the learning rate.
        train_dataloader (:obj:`torch.utils.data.dataloader.DataLoader`, `optional`):
            The current dataloader used for training.
        eval_dataloader (:obj:`torch.utils.data.dataloader.DataLoader`, `optional`):
            The current dataloader used for training.
        metrics (:obj:`Dict[str, float]`):
            The metrics computed by the last evaluation phase.

            Those are only accessible in the event :obj:`on_evaluate`.
        logs  (:obj:`Dict[str, float]`):
            The values to log.

            Those are only accessible in the event :obj:`on_log`.

    The :obj:`control` object is the only one that can be changed by the callback, in which case the event that changes
    it should return the modified version.

    The argument :obj:`args`, :obj:`state` and :obj:`control` are positionals for all events, all the others are
    grouped in :obj:`kwargs`. You can unpack the ones you need in the signature of the event using them. As an example,
    see the code of the simple :class:`~transformer.PrinterCallback`.

    Example::

        class PrinterCallback(TrainerCallback):

            def on_log(self, args, state, control, logs=None, **kwargs):
                _ = logs.pop("total_flos", None)
                if state.is_local_process_zero:
                    print(logs)
    """
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the :class:`~transformers.Trainer`.
        """
        ...
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
        ...
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        ...
    
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        ...
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        ...
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        ...
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        ...
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        ...
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        ...
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """
        ...
    
    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """
        ...
    


class CallbackHandler(TrainerCallback):
    """Internal class that just calls the list of callbacks in order."""
    def __init__(self, callbacks, model, tokenizer, optimizer, lr_scheduler) -> None:
        ...
    
    def add_callback(self, callback):
        ...
    
    def pop_callback(self, callback):
        ...
    
    def remove_callback(self, callback):
        ...
    
    @property
    def callback_list(self):
        ...
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        ...
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        ...
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        ...
    
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        ...
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        ...
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        ...
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        ...
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        ...
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        ...
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs):
        ...
    
    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        ...
    
    def call_event(self, event, args, state, control, **kwargs):
        ...
    


class DefaultFlowCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that handles the default flow of the training loop for logs, evaluation
    and checkpoints.
    """
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        ...
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        ...
    


class ProgressCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that displays the progress of training or evaluation.
    """
    def __init__(self) -> None:
        ...
    
    def on_train_begin(self, args, state, control, **kwargs):
        ...
    
    def on_step_end(self, args, state, control, **kwargs):
        ...
    
    def on_prediction_step(self, args, state, control, eval_dataloader=..., **kwargs):
        ...
    
    def on_evaluate(self, args, state, control, **kwargs):
        ...
    
    def on_log(self, args, state, control, logs=..., **kwargs):
        ...
    
    def on_train_end(self, args, state, control, **kwargs):
        ...
    


class PrinterCallback(TrainerCallback):
    """
    A bare :class:`~transformers.TrainerCallback` that just prints the logs.
    """
    def on_log(self, args, state, control, logs=..., **kwargs):
        ...
    


class EarlyStoppingCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that handles early stopping.

    Args:
       early_stopping_patience (:obj:`int`):
            Use with :obj:`metric_for_best_model` to stop training when the specified metric worsens for
            :obj:`early_stopping_patience` evaluation calls.
       early_stopping_threshold(:obj:`float`, `optional`):
            Use with TrainingArguments :obj:`metric_for_best_model` and :obj:`early_stopping_patience` to denote how
            much the specified metric must improve to satisfy early stopping conditions. `

    This callback depends on :class:`~transformers.TrainingArguments` argument `load_best_model_at_end` functionality
    to set best_metric in :class:`~transformers.TrainerState`.
    """
    def __init__(self, early_stopping_patience: int = ..., early_stopping_threshold: Optional[float] = ...) -> None:
        ...
    
    def check_metric_value(self, args, state, control, metric_value):
        ...
    
    def on_train_begin(self, args, state, control, **kwargs):
        ...
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        ...
    


