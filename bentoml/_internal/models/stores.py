import logging
import os
import typing as t

from simple_di import Provide, inject

from ..configuration.containers import BentoMLContainer
from ..models.base import Model
from ..types import PathType

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
                ", and my_nlp_model:20210907084629_36AE55). Default to `latest`"
            )
            version = "latest"
        return name, version


class LocalModelStore:
    @inject
    def __init__(self, base_dir: PathType = Provide[BentoMLContainer.bentoml_home]):
        self._BASE_DIR = os.path.join(base_dir, LOCAL_MODELSTORE_NAMESPACE)

    def list_model(self, name: t.Optional[str] = None) -> t.List[str]:
        """
        bentoml models list -> t.List[models name under BENTOML_HOME/models]
        bentoml models list my_nlp_models -> t.List[model_version]
        """
        if not name:
            path = self._BASE_DIR
        else:
            name, version = _process_name(name)
            path = os.path.join(self._BASE_DIR, name)
        return [_f.name for _f in os.scandir(path)]

    def get_model(self, name: str, model_instance: "_MT", kwargs):
        name, version = _process_name(name)

    def delete_model(self, name: str):
        ...

    def push_model(self, name: str):
        ...

    def pull_model(self, name: str):
        ...

    def export_model(self, name: str):
        ...

    def import_model(self, name: str):
        ...

    def _create_model_stores(self, name: str, model_instance: "_MT"):
        ...


# Global modelstore instance
modelstore = LocalModelStore()
