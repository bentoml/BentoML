from .model import Model
from .model import copy_model
from .model import ModelStore

EXPORTED_STORE_PREFIX = "exported"
SAVE_NAMESPACE = "saved_model"
MODEL_YAML_NAMESPACE = "model_details"

H5_EXT = ".h5"
HDF5_EXT = ".hdf5"
JSON_EXT = ".json"
PKL_EXT = ".pkl"
PTH_EXT = ".pth"
PT_EXT = ".pt"
TXT_EXT = ".txt"
YAML_EXT = ".yaml"
MODEL_EXT = ".model"

__all__ = ["Model", "ModelStore", "copy_model"]
