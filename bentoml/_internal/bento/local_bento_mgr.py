from typing import List, Optional

from simple_di import Provide, inject

from ..configuration.containers import BentoMLContainer
from ..types import BentoTag, PathType


class LocalBentoMgr:
    @inject
    def __init__(self, base_dir: PathType = Provide[BentoMLContainer.bentoml_home]):
        self.base_dir = base_dir

    def list_bento(self, name: Optional[str] = None) -> List[str]:
        pass

    def get_bento(self, tag: BentoTag):
        pass

    def export_bento(self, tag: BentoTag, file_path: PathType):
        pass

    def import_bento(self, file_path: PathType):
        pass

    def delete_bento(self, tags: List[BentoTag]):
        pass

    def push_bento(self, tag: BentoTag):
        pass

    def pull_bento(self, tag: BentoTag):
        pass

    def _add_bento(self, tag: BentoTag, tmp_bento_path: PathType):
        pass
