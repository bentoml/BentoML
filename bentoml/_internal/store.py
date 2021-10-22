import typing as t
from contextlib import contextmanager

import fs
from fs.base import FS

from ..exceptions import BentoMLException
from .types import BentoTag, PathType

SUPPORTED_COMPRESSION_TYPE = [".gz"]


class Store:
    """An FsStore manages items under the given base filesystem.

    Note that FsStore has no consistency checks; it assumes that no direct modification
    of the files in its directory has occurred.

    """

    def __init__(self, base_path: PathType):
        self.fs = fs.open_fs(str(base_path))

    def push(self, tag: str) -> None:
        ...

    def pull(self, tag: str) -> None:
        ...

    def list(self, tag: t.Optional[t.Union[BentoTag, str]] = None) -> t.List[BentoTag]:
        if not tag:
            return sorted(
                [ver for _d in self.fs.listdir("/") for ver in self.list(_d)],
                key=str,
            )

        _tag = BentoTag.from_taglike(tag)
        if _tag.version is None:
            return [
                BentoTag(_tag.name, f.name)
                for f in self.fs.scandir(_tag.name)
                if not f.islink()
            ]
        else:
            path = fs.path.combine(_tag.name, _tag.version)
            if _tag.version == "latest":
                _tag.version = self.fs.readtext(path)
                path = fs.path.combine(_tag.name, _tag.version)
            return sorted(list(self.fs.walk(path)), key=str)

    def get(self, tag: t.Union[BentoTag, str]) -> FS:
        """
        store.get("my_bento")
        store.get("my_bento:v1.0.0")
        store.get(BentoTag("my_bento", "latest"))
        """
        _tag = BentoTag.from_taglike(tag)
        if _tag.version is None:
            _tag.version = self.fs.readtext(fs.path.combine(_tag.name, "latest"))
        path = _tag.path()
        if not self.fs.exists(path):
            raise FileNotFoundError(
                f"Item '{tag}' is not found in BentoML store {self.fs}."
            )
        return self.fs.opendir(path)

    @contextmanager
    def register(self, tag: t.Union[str, BentoTag]):
        _tag = BentoTag.from_taglike(tag)

        item_path = _tag.path()
        if self.fs.exists(item_path):
            raise BentoMLException(
                f"Item '{_tag}' already exists in the store {self.fs}"
            )
        self.fs.makedirs(item_path)
        try:
            yield self.fs.getsyspath(item_path)
        finally:
            # item generation is most likely successful, link latest path
            latest_path = fs.path.combine(_tag.name, "latest")
            with self.fs.open(latest_path, "w") as latest_file:
                latest_file.write(_tag.version)

    def delete(self, tag: t.Union[str, BentoTag], skip_confirm: bool = True) -> None:
        if not skip_confirm:
            raise BentoMLException(
                f"'skip_confirm={skip_confirm}'; not deleting {tag}. If you want to bypass this check change 'skip_confirm=True'"
            )

        _tag = BentoTag.from_taglike(tag)
        if _tag.version is None:
            self.fs.removetree()
            return

        path = _tag.path()
        self.fs.removetree(path)
        new_latest = sorted(self.fs.scandir(_tag.name), key=lambda f: f.created)[0]
        latest_path = fs.path.combine(_tag.name, "latest")
        with self.fs.open(latest_path, "w") as latest_file:
            latest_file.write(new_latest.name)
