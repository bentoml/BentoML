

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..file_utils import add_end_docstrings
from ..modelcard import ModelCard
from ..modeling_tf_utils import TFPreTrainedModel
from ..modeling_utils import PreTrainedModel
from ..models.auto.configuration_auto import AutoConfig
from ..tokenization_utils import PreTrainedTokenizer

def infer_framework_load_model(model, config: AutoConfig, model_classes: Optional[Dict[str, Tuple[type]]] = ..., task: Optional[str] = ..., framework: Optional[str] = ..., **model_kwargs):
    """
    Select framework (TensorFlow or PyTorch) to use from the :obj:`model` passed. Returns a tuple (framework, model).

    If :obj:`model` is instantiated, this function will just infer the framework from the model class. Otherwise
    :obj:`model` is actually a checkpoint name and this method will try to instantiate it using :obj:`model_classes`.
    Since we don't want to instantiate the model twice, this model is returned for use by the pipeline.

    If both frameworks are installed and available for :obj:`model`, PyTorch is selected.

    Args:
        model (:obj:`str`, :class:`~transformers.PreTrainedModel` or :class:`~transformers.TFPreTrainedModel`):
            The model to infer the framework from. If :obj:`str`, a checkpoint name. The model to infer the framewrok
            from.
        config (:class:`~transformers.AutoConfig`):
            The config associated with the model to help using the correct class
        model_classes (dictionary :obj:`str` to :obj:`type`, `optional`):
            A mapping framework to class.
        task (:obj:`str`):
            The task defining which pipeline will be returned.
        model_kwargs:
            Additional dictionary of keyword arguments passed along to the model's :obj:`from_pretrained(...,
            **model_kwargs)` function.

    Returns:
        :obj:`Tuple`: A tuple framework, model.
    """
    ...

def infer_framework_from_model(model, model_classes: Optional[Dict[str, Tuple[type]]] = ..., task: Optional[str] = ..., framework: Optional[str] = ..., **model_kwargs):
    """
    Select framework (TensorFlow or PyTorch) to use from the :obj:`model` passed. Returns a tuple (framework, model).

    If :obj:`model` is instantiated, this function will just infer the framework from the model class. Otherwise
    :obj:`model` is actually a checkpoint name and this method will try to instantiate it using :obj:`model_classes`.
    Since we don't want to instantiate the model twice, this model is returned for use by the pipeline.

    If both frameworks are installed and available for :obj:`model`, PyTorch is selected.

    Args:
        model (:obj:`str`, :class:`~transformers.PreTrainedModel` or :class:`~transformers.TFPreTrainedModel`):
            The model to infer the framework from. If :obj:`str`, a checkpoint name. The model to infer the framewrok
            from.
        model_classes (dictionary :obj:`str` to :obj:`type`, `optional`):
            A mapping framework to class.
        task (:obj:`str`):
            The task defining which pipeline will be returned.
        model_kwargs:
            Additional dictionary of keyword arguments passed along to the model's :obj:`from_pretrained(...,
            **model_kwargs)` function.

    Returns:
        :obj:`Tuple`: A tuple framework, model.
    """
    ...

def get_framework(model, revision: Optional[str] = ...):
    """
    Select framework (TensorFlow or PyTorch) to use.

    Args:
        model (:obj:`str`, :class:`~transformers.PreTrainedModel` or :class:`~transformers.TFPreTrainedModel`):
            If both frameworks are installed, picks the one corresponding to the model passed (either a model class or
            the model name). If no specific model is provided, defaults to using PyTorch.
    """
    ...

def get_default_model(targeted_task: Dict, framework: Optional[str], task_options: Optional[Any]) -> str:
    """
    Select a default model to use for a given task. Defaults to pytorch if ambiguous.

    Args:
        targeted_task (:obj:`Dict` ):
           Dictionary representing the given task, that should contain default models

        framework (:obj:`str`, None)
           "pt", "tf" or None, representing a specific framework if it was specified, or None if we don't know yet.

        task_options (:obj:`Any`, None)
           Any further value required by the task to get fully specified, for instance (SRC, TGT) languages for
           translation task.

    Returns

        :obj:`str` The model string representing the default model for this pipeline
    """
    ...

class PipelineException(Exception):
    """
    Raised by a :class:`~transformers.Pipeline` when handling __call__.

    Args:
        task (:obj:`str`): The task of the pipeline.
        model (:obj:`str`): The model used by the pipeline.
        reason (:obj:`str`): The error message to display.
    """
    def __init__(self, task: str, model: str, reason: str) -> None:
        ...
    


class ArgumentHandler(ABC):
    """
    Base interface for handling arguments for each :class:`~transformers.pipelines.Pipeline`.
    """
    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...
    


class PipelineDataFormat:
    """
    Base class for all the pipeline supported data format both for reading and writing. Supported data formats
    currently includes:

    - JSON
    - CSV
    - stdin/stdout (pipe)

    :obj:`PipelineDataFormat` also includes some utilities to work with multi-columns like mapping from datasets
    columns to pipelines keyword arguments through the :obj:`dataset_kwarg_1=dataset_column_1` format.

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
    """
    SUPPORTED_FORMATS = ...
    def __init__(self, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite: bool = ...) -> None:
        ...
    
    @abstractmethod
    def __iter__(self):
        ...
    
    @abstractmethod
    def save(self, data: Union[dict, List[dict]]):
        """
        Save the provided data object with the representation for the current
        :class:`~transformers.pipelines.PipelineDataFormat`.

        Args:
            data (:obj:`dict` or list of :obj:`dict`): The data to store.
        """
        ...
    
    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        """
        Save the provided data object as a pickle-formatted binary data on the disk.

        Args:
            data (:obj:`dict` or list of :obj:`dict`): The data to store.

        Returns:
            :obj:`str`: Path where the data has been saved.
        """
        ...
    
    @staticmethod
    def from_str(format: str, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=...) -> PipelineDataFormat:
        """
        Creates an instance of the right subclass of :class:`~transformers.pipelines.PipelineDataFormat` depending on
        :obj:`format`.

        Args:
            format: (:obj:`str`):
                The format of the desired pipeline. Acceptable values are :obj:`"json"`, :obj:`"csv"` or :obj:`"pipe"`.
            output_path (:obj:`str`, `optional`):
                Where to save the outgoing data.
            input_path (:obj:`str`, `optional`):
                Where to look for the input data.
            column (:obj:`str`, `optional`):
                The column to read.
            overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to overwrite the :obj:`output_path`.

        Returns:
            :class:`~transformers.pipelines.PipelineDataFormat`: The proper data format.
        """
        ...
    


class CsvPipelineDataFormat(PipelineDataFormat):
    """
    Support for pipelines using CSV data format.

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
    """
    def __init__(self, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=...) -> None:
        ...
    
    def __iter__(self):
        ...
    
    def save(self, data: List[dict]):
        """
        Save the provided data object with the representation for the current
        :class:`~transformers.pipelines.PipelineDataFormat`.

        Args:
            data (:obj:`List[dict]`): The data to store.
        """
        ...
    


class JsonPipelineDataFormat(PipelineDataFormat):
    """
    Support for pipelines using JSON file format.

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
    """
    def __init__(self, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=...) -> None:
        ...
    
    def __iter__(self):
        ...
    
    def save(self, data: dict):
        """
        Save the provided data object in a json file.

        Args:
            data (:obj:`dict`): The data to store.
        """
        ...
    


class PipedPipelineDataFormat(PipelineDataFormat):
    """
    Read data from piped input to the python process. For multi columns data, columns should separated by \t

    If columns are provided, then the output will be a dictionary with {column_x: value_x}

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
    """
    def __iter__(self):
        ...
    
    def save(self, data: dict):
        """
        Print the data.

        Args:
            data (:obj:`dict`): The data to store.
        """
        ...
    
    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        ...
    


class _ScikitCompat(ABC):
    """
    Interface layer for the Scikit and Keras compatibility.
    """
    @abstractmethod
    def transform(self, X):
        ...
    
    @abstractmethod
    def predict(self, X):
        ...
    


PIPELINE_INIT_ARGS: str = ...
class Pipeline(_ScikitCompat):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
    operations:

        Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

    Pipeline supports running on CPU or GPU through the device argument (see below).

    Some pipeline, like for instance :class:`~transformers.FeatureExtractionPipeline` (:obj:`'feature-extraction'` )
    output large tensor object as nested-lists. In order to avoid dumping such large structure as textual data we
    provide the :obj:`binary_output` constructor argument. If set to :obj:`True`, the output will be stored in the
    pickle format.
    """
    default_input_names: None = ...
    model: Union[PreTrainedModel, TFPreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizer]
    feature_extractor: Optional[PreTrainedFeatureExtractor]
    def __init__(self, model: Union[PreTrainedModel, TFPreTrainedModel], tokenizer: Optional[PreTrainedTokenizer] = ..., feature_extractor: Optional[PreTrainedFeatureExtractor] = ..., modelcard: Optional[ModelCard] = ..., framework: Optional[str] = ..., task: str = ..., args_parser: ArgumentHandler = ..., device: int = ..., binary_output: bool = ...) -> None:
        ...
    
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the pipeline's model and tokenizer.

        Args:
            save_directory (:obj:`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
        """
        ...
    
    def transform(self, X: Any) -> Any:
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        ...
    
    def predict(self, X: Any) -> Any:
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        ...
    
    @contextmanager
    def device_placement(self) -> None:
        """
        Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.

        Returns:
            Context manager

        Examples::

            # Explicitly ask for tensor allocation on CUDA device :0
            pipe = pipeline(..., device=0)
            with pipe.device_placement():
                # Every framework specific tensor allocation will be done on the request device
                output = pipe(...)
        """
        ...
    
    def ensure_tensor_on_device(self, **inputs: Any) -> None:
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be :obj:`torch.Tensor`): The tensors to place on :obj:`self.device`.

        Return:
            :obj:`Dict[str, torch.Tensor]`: The same as :obj:`inputs` but on the proper device.
        """
        ...
    
    def check_model_type(self, supported_models: Union[List[str], Dict[str, Any]]) -> bool:
        """
        Check if the model class is in supported by the pipeline.

        Args:
            supported_models (:obj:`List[str]` or :obj:`dict`):
                The list of models supported by the pipeline, or a dictionary with model class values.
        """
        ...
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...
    


