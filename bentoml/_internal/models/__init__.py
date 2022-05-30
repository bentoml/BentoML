from .model import Model
from .model import copy_model
from .model import ModelStore
from .model import ModelContext
from .model import ModelOptions

# Deprecated. Use framework module local constants and name the saved files with API
# Version in mind. E.g.:
# api_v1_model_file_name = "saved_model.pkl"
# api_v2_model_file_name = "torch_model.pth"
SAVE_NAMESPACE = "saved_model"
JSON_EXT = ".json"
PKL_EXT = ".pkl"
PTH_EXT = ".pth"
TXT_EXT = ".txt"
YAML_EXT = ".yaml"

__all__ = ["Model", "ModelStore", "ModelContext", "ModelOptions", "copy_model"]
