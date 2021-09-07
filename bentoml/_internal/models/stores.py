import logging
import typing as t
from pathlib import Path

from simple_di import Provide, inject

from ..configuration.containers import BentoMLContainer
from ..models.base import Model, _validate_or_create_dir
from ..types import PathType
from ..utils import generate_new_version_id

LOCAL_MODELSTORE_NAMESPACE = "models"
_MT = t.TypeVar("_MT", bound=Model)

logger = logging.getLogger(__name__)


def _process_name(model_name: str, sep: str = ":") -> t.Tuple[str, str]:
    # name can be my_model, my_model:latest, my_model:20210907084629_36AE55
    if sep not in model_name:
        return model_name, "latest"
    else:
        name, version = model_name.split(sep)
        if not version:
            # in case users define name in format `my_nlp_model:`
            logger.warning(
                f"{model_name} contains leading `:`. Make sure to "
                "have correct name format (my_nlp_model, my_nlp_model:latest"
                ", and my_nlp_model:20210907_36AE55). Assuming defaults to"
                " `latest`..."
            )
            version = "latest"
        return name, version


class LocalModelStore:
    @inject
    def __init__(self, base_dir: PathType = Provide[BentoMLContainer.bentoml_home]):
        self._BASE_DIR = Path(base_dir, LOCAL_MODELSTORE_NAMESPACE)
        _validate_or_create_dir(self._BASE_DIR)

    def list_model(self, name: t.Optional[str] = None) -> t.List[str]:
        """
        bentoml models list -> t.List[models name under BENTOML_HOME/models]
        bentoml models list my_nlp_models -> t.List[model_version]
        """
        if not name:
            path = self._BASE_DIR
        else:
            name, version = _process_name(name)
            path = Path(self._BASE_DIR, name)
        return [_f.name for _f in path.iterdir()]

    def save_model(self, name: str, model_instance: "_MT") -> str:
        """
        bentoml.pytorch.save("my_nlp_model", MyTorchModel()) -> name:version
        """
        version = generate_new_version_id()
        path = Path(self._BASE_DIR, name, version)
        self._create_stores(path, model_instance)
        path.symlink_to(Path(self._BASE_DIR, name, "latest"))
        return f"{name}:{version}"

    def get_model(self, name: str, model_instance: "_MT", **load_kwargs) -> t.Any:
        """
        bentoml.pytorch.load("my_nlp_model")
        """
        name, version = _process_name(name)
        path = Path(self._BASE_DIR, name, version)
        if not path.exists():
            raise FileNotFoundError(
                "Given model name is not found in BentoML model stores. "
                "Make sure to have the correct model name."
            )
        else:
            return model_instance.load(path, **load_kwargs)

    def delete_model(self, name: str, skip_confirm: bool = False):
        """
        bentoml models delete
        """
        basepath = Path(self._BASE_DIR, name)
        if ":" not in name:
            basepath.rmdir()
        else:
            name, version = _process_name(name)
            if version == "latest":
                # covers cases where users mistype
                # leading ":"
                basepath.rmdir()
            else:
                path = Path(basepath, version)
                path.rmdir()
                path.unlink(missing_ok=True)

    def push_model(self, name: str):
        ...

    def pull_model(self, name: str):
        ...

    def export_model(self, name: str):
        ...

    def import_model(self, name: str):
        ...

    @staticmethod
    def _create_stores(path: PathType, model_instance: "_MT"):
        Path(path).mkdir(parents=True)
        model_instance.save(path)


# Global modelstore instance
modelstore = LocalModelStore()
