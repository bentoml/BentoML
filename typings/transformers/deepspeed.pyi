

from .file_utils import is_torch_available

if is_torch_available():
    ...
logger = ...
def is_deepspeed_available():
    ...

class HfDeepSpeedConfig:
    """
    This object contains a DeepSpeed configuration dictionary and can be quickly queried for things like zero stage.

    A ``weakref`` of this object is stored in the module's globals to be able to access the config from areas where
    things like the Trainer object is not available (e.g. ``from_pretrained`` and ``_get_resized_embeddings``).
    Therefore it's important that this object remains alive while the program is still running.

    :class:`~transformers.Trainer` uses the ``HfTrainerDeepSpeedConfig`` subclass instead. That subclass has logic to
    sync the configuration with values of :class:`~transformers.TrainingArguments` by replacing special placeholder
    values: ``"auto"``. Without this special logic the DeepSpeed configuration is not modified in any way.

    Args:
        config_file_or_dict (:obj:`Union[str, Dict]`) - path to DeepSpeed config file or dict.

    """
    def __init__(self, config_file_or_dict) -> None:
        ...
    
    def find_config_node(self, ds_key_long):
        ...
    
    def get_value(self, ds_key_long, default=...):
        """
        Returns the set value or ``default`` if no value is set
        """
        ...
    
    def is_true(self, ds_key_long):
        """
        Returns :obj:`True`/:obj:`False` only if the value is set, always :obj:`False` otherwise. So use this method to
        ask the very specific question of whether the value is set to :obj:`True` (and it's not set to :obj:`False` or
        isn't set).

        """
        ...
    
    def is_false(self, ds_key_long):
        """
        Returns :obj:`True`/:obj:`False` only if the value is set, always :obj:`False` otherwise. So use this method to
        ask the very specific question of whether the value is set to :obj:`False` (and it's not set to :obj:`True` or
        isn't set).
        """
        ...
    
    def is_zero2(self):
        ...
    
    def is_zero3(self):
        ...
    
    def is_offload(self):
        ...
    


class HfTrainerDeepSpeedConfig(HfDeepSpeedConfig):
    """
    The ``HfTrainerDeepSpeedConfig`` object is meant to be created during ``TrainingArguments`` object creation and has
    the same lifespan as the latter.
    """
    def __init__(self, config_file_or_dict) -> None:
        ...
    
    def dtype(self):
        ...
    
    def fill_match(self, ds_key_long, hf_val, hf_key=..., must_match=...):
        """
        A utility method that massages the config file and can optionally verify that the values match.

        1. Replace "auto" values with ``TrainingArguments`` value.

        2. If it wasn't "auto" and ``must_match`` is true, then check that DS config matches Trainer
        config values and if mismatched add the entry to ``self.mismatched`` - will assert during
        ``trainer_config_finalize`` for one or more mismatches.

        """
        ...
    
    fill_only = ...
    def trainer_config_process(self, args):
        """
        Adjust the config with ``TrainingArguments`` values. This stage is run during ``TrainingArguments`` object
        creation.
        """
        ...
    
    def trainer_config_finalize(self, args, model, num_training_steps):
        """
        This stage is run after we have the model and know num_training_steps.

        Now we we can complete the configuration process.
        """
        ...
    


_hf_deepspeed_config_weak_ref = ...
def set_hf_deepspeed_config(hf_deepspeed_config_obj):
    ...

def is_deepspeed_zero3_enabled():
    ...

def deepspeed_config():
    ...

def deepspeed_init(trainer, num_training_steps, resume_from_checkpoint=...):
    """
    Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.

    If ``resume_from_checkpoint`` was passed then an attempt to resume from a previously saved checkpoint will be made.

    Args:
        trainer: Trainer object
        num_training_steps: per single gpu
        resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load

    Returns: model, optimizer, lr_scheduler

    """
    ...

