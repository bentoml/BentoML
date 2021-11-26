import os
from collections import OrderedDict
from typing import Any, List, Union
from ...configuration_utils import PretrainedConfig
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...modeling_tf_utils import TFPreTrainedModel
from ...modeling_utils import PreTrainedModel

PathLike = Union[str, os.PathLike[str]]

class _BaseAutoModelClass:
    _model_mapping: OrderedDict[str, Any] = ...
    @classmethod
    def from_config(
        cls, config: PretrainedConfig, **kwargs: Any
    ) -> Union[PreTrainedModel, TFPreTrainedModel, FlaxPreTrainedModel]: ...
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: PathLike, *model_args: Any, **kwargs: Any
    ) -> Union[PreTrainedModel, FlaxPreTrainedModel, TFPreTrainedModel]: ...

def insert_head_doc(docstring: str, head_doc: str = ...) -> str: ...
def auto_class_update(
    cls: _BaseAutoModelClass, checkpoint_for_example: str = ..., head_doc: str = ...
) -> _BaseAutoModelClass: ...
def get_values(model_mapping: OrderedDict[str, Any]) -> List[Any]: ...
