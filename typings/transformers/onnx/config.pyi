

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from transformers import PretrainedConfig, PreTrainedTokenizer, TensorType

DEFAULT_ONNX_OPSET = ...
EXTERNAL_DATA_FORMAT_SIZE_LIMIT = ...
@dataclasses.dataclass
class PatchingSpec:
    """
    Data class that holds patching specifications.

    Args:
        o: Module / object where the op to patch is located
        name: Name of the op to monkey patch
        custom_op: Custom op that patches the original op
        orig_op: Original op that is being patched
        op_wrapper: Wrapper (optional) that wraps both the original and custom ops.
            It is useful for ops that are class or static methods for instance.
    """
    o: Any
    name: str
    custom_op: Callable
    orig_op: Optional[Callable] = ...
    op_wrapper: Optional[Callable] = ...


class OnnxConfig(ABC):
    """
    Base class for ONNX exportable model describing metadata on how to export the model through the ONNX format.
    """
    DEFAULT_FIXED_BATCH = ...
    DEFAULT_FIXED_SEQUENCE = ...
    _TASKS_TO_COMMON_OUTPUTS = ...
    def __init__(self, config: PretrainedConfig, task: str = ..., patching_specs: List[PatchingSpec] = ...) -> None:
        ...
    
    @classmethod
    def from_model_config(cls, config: PretrainedConfig, task: str = ...) -> OnnxConfig:
        """
        Instantiate a OnnxConfig for a specific model

        Args:
            config: The model's configuration to use when exporting to ONNX

        Returns:
            OnnxConfig for this model
        """
        ...
    
    @property
    @abstractmethod
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the input tensors to provide to the model

        Returns:
            For each input: its name associated to the axes symbolic name and the axis position within the tensor
        """
        ...
    
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the output tensors to provide to the model

        Returns:
            For each output: its name associated to the axes symbolic name and the axis position within the tensor
        """
        ...
    
    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        """
        Dictionary of keys to override in the model's config before exporting

        Returns:
            Dictionary with the keys (and their corresponding values) to override
        """
        ...
    
    @property
    def default_batch_size(self) -> int:
        """
        The default batch size to use if no other indication

        Returns:
            Integer > 0
        """
        ...
    
    @property
    def default_sequence_length(self) -> int:
        """
        The default sequence length to use if no other indication

        Returns:
            Integer > 0
        """
        ...
    
    @property
    def default_onnx_opset(self) -> int:
        """
        Which onnx opset to use when exporting the model

        Returns:
            Integer ONNX Opset version
        """
        ...
    
    @staticmethod
    def use_external_data_format(num_parameters: int) -> bool:
        """
        Flag indicating if the model requires using external data format

        Args:
            num_parameters: Number of parameter on the model

        Returns:
            True if model.num_parameters() * size_of(float32) >= 2Gb False otherwise
        """
        ...
    
    def generate_dummy_inputs(self, tokenizer: PreTrainedTokenizer, batch_size: int = ..., seq_length: int = ..., is_pair: bool = ..., framework: Optional[TensorType] = ...) -> Mapping[str, Any]:
        """
        Generate inputs to provide to the ONNX exporter for the specific framework

        Args:
            tokenizer: The tokenizer associated with this model configuration
            batch_size: The batch size (int) to export the model for (-1 means dynamic axis)
            seq_length: The sequence length (int) to export the model for (-1 means dynamic axis)
            is_pair: Indicate if the input is a pair (sentence 1, sentence 2)
            framework: The framework (optional) the tokenizer will generate tensor for

        Returns:
            Mapping[str, Tensor] holding the kwargs to provide to the model's forward function
        """
        ...
    
    def patch_ops(self):
        ...
    
    def restore_ops(self):
        ...
    
    @staticmethod
    def flatten_output_collection_property(name: str, field: Iterable[Any]) -> Dict[str, Any]:
        """
        Flatten any potential nested structure expanding the name of the field with the index of the element within the
        structure.

        Args:
            name: The name of the nested structure
            field: The structure to, potentially, be flattened

        Returns:
            (Dict[str, Any]): Outputs with flattened structure and key mapping this new structure.

        """
        ...
    


class OnnxConfigWithPast(OnnxConfig, ABC):
    def __init__(self, config: PretrainedConfig, task: str = ..., patching_specs: List[PatchingSpec] = ..., use_past: bool = ...) -> None:
        ...
    
    @classmethod
    def with_past(cls, config: PretrainedConfig, task: str = ...) -> OnnxConfigWithPast:
        """
        Instantiate a OnnxConfig with `use_past` attribute set to True

        Args:
            config: The underlying model's config to use when exporting to ONNX

        Returns:
            OnnxConfig with `.use_past = True`
        """
        ...
    
    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        ...
    
    def generate_dummy_inputs(self, tokenizer: PreTrainedTokenizer, batch_size: int = ..., seq_length: int = ..., is_pair: bool = ..., framework: Optional[TensorType] = ...) -> Mapping[str, Any]:
        ...
    
    @staticmethod
    def flatten_output_collection_property(name: str, field: Iterable[Any]) -> Dict[str, Any]:
        ...
    


