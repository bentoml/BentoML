import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from simple_di import Provide, inject

from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.types import BentoTag, PathType
from bentoml.exceptions import BentoMLException

logger = logging.getLogger(__name__)

BENTO_STORE_PREFIX = "bentos"


class LocalBentoStore:
    @inject
    def __init__(self, bentoml_home: PathType = Provide[BentoMLContainer.bentoml_home]):
        self.base_dir = os.path.join(bentoml_home, BENTO_STORE_PREFIX)
        if not os.path.isdir(self.base_dir):
            Path(self.base_dir).mkdir(parents=True)

    def list_bento(self, name: Optional[str] = None) -> List[str]:
        pass

    def get_bento(self, tag: str):
        pass

    def export_bento(self, tag: str, file_path: PathType):
        pass

    def import_bento(self, file_path: PathType):
        pass

    def delete_bento(self, tags: List[str]):
        pass

    def push_bento(self, tag: str):
        pass

    def pull_bento(self, tag: str):
        pass

    @contextmanager
    def register_bento(self, tag: str):
        try:
            bento_tag = BentoTag.from_str(tag)
            bento_path = self._create_bento_path(bento_tag)
            yield bento_path
        finally:
            if not os.path.isfile(os.path.join(bento_path, "bento.yaml")):
                # Build has failed
                logger.warning(
                    f"Failed creating Bento file for {tag}, deleting {bento_path}"
                )
                shutil.rmtree(bento_path)
            else:
                # Build is most likely successful, link latest bento path
                latest_path = Path(self.base_dir, bento_tag.name, "latest")
                if latest_path.is_symlink():
                    latest_path.unlink()
                latest_path.symlink_to(bento_path)

    def _create_bento_path(self, tag: BentoTag):
        bento_path = os.path.join(self.base_dir, tag.name, tag.version)
        if os.path.isdir(bento_path):
            raise BentoMLException(f"Bento path {bento_path} already exist")
        Path(bento_path).mkdir(parents=True)
        return bento_path


bento_store = LocalBentoStore()
