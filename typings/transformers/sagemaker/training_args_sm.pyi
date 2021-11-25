

from dataclasses import dataclass

from transformers.training_args import TrainingArguments

logger = ...
def is_sagemaker_model_parallel_available():
    ...

if is_sagemaker_model_parallel_available():
    ...
@dataclass
class SageMakerTrainingArguments(TrainingArguments):
    mp_parameters: str = ...
    def __post_init__(self):
        ...
    
    @property
    def world_size(self):
        ...
    
    @property
    def place_model_on_device(self):
        ...
    


