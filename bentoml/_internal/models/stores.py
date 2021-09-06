import os

from simple_di import Provide, inject

import bentoml._internal.constants as _const

from ..configuration.containers import BentoMLContainer
from ..types import PathType


class LocalModelStore:
    @inject
    def __init__(self, base_dir: PathType = Provide[BentoMLContainer.bentoml_home]):
        self._base_dir = os.path.join(base_dir, _const.LOCAL_MODELSTORE_NAMESPACE)

    def list_model(self):
        ...

    def get_model(self, name: str):
        ...

    def delete_model(self):
        ...

    def push_model(self):
        ...

    def pull_model(self):
        ...

    def export_model(self):
        ...

    def import_model(self):
        ...

    def _create_model(self, name: str):
        ...
