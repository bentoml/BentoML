MODEL_STORE_PREFIX = "models"
EXPORTED_STORE_PREFIX = "exported"
SAVE_NAMESPACE = "saved_model"
MODEL_YAML_NAMESPACE = "model_details"

SAVE_INIT_DOCS = """\
Save a model instance to BentoML modelstore.

Examples::
    # train.py
    model = MyPyTorchModel().train()  # type: torch.nn.Module
    ...
    import bentoml.pytorch
    semver = bentoml.pytorch.save("my_nlp_model", model, embedding=128)

Args:
    name (`str`):
        Name for given model instance. This should pass Python identifier check.
    metadata (`~bentoml._internal.types.GenericDictType`):
        Custom metadata for given model."""

SAVE_RETURNS_DOCS = """\

Returns:
    store_name (`str` with a format `name:generated_id`) where `name` is the defined name
    user set for their models, and `generated_id` will be generated UUID by BentoML."""

LOAD_INIT_DOCS = """\
Load a model from BentoML modelstore with given name.

Args:
    name (`str`):
        Name of a saved model in BentoML modelstore."""

H5_EXT = ".h5"
HDF5_EXT = ".hdf5"
JSON_EXT = ".json"
PKL_EXT = ".pkl"
PTH_EXT = ".pth"
PT_EXT = ".pt"
TXT_EXT = ".txt"
YAML_EXT = ".yaml"
