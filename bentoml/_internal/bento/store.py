import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from ..types import BentoTag, PathType
from ..utils import validate_or_create_dir

logger = logging.getLogger(__name__)


class BentoStore:
    """A BentoStore manage bentos under the given base_dir.

    Note that BentoStore is designed to rely on just the file system itself. It assumes
    that no direct modification of files could be made for anything under the base_dir
    """

    def __init__(self, base_dir: PathType):
        self._base_dir = base_dir
        validate_or_create_dir(self._base_dir)

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
    def register_bento(self, tag: BentoTag):
        try:
            bento_path = self._create_bento_path(tag)
            yield bento_path
        finally:
            # Build is most likely successful, link latest bento path
            latest_path = Path(self._base_dir, tag.name, "latest")
            if latest_path.is_symlink():
                latest_path.unlink()
            latest_path.symlink_to(bento_path)

    def _create_bento_path(self, tag: BentoTag) -> "Path":
        bento_path = Path(self._base_dir, tag.name, tag.version)
        validate_or_create_dir(bento_path)
        return bento_path
