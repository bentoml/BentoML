import datetime
import typing as t
from abc import ABC, abstractmethod
from contextlib import contextmanager

import fs
from fs.base import FS

from ..exceptions import BentoMLException
from .types import PathType, Tag

T = t.TypeVar("T")


class StoreItem(ABC):
    @classmethod
    @abstractmethod
    def from_fs(CLS: t.Type[T], tag: Tag, fs: FS) -> T:
        pass

    @abstractmethod
    def creation_time(self) -> datetime.datetime:
        pass


Item = t.TypeVar("Item", bound=StoreItem)


class Store(ABC, t.Generic[Item]):
    """An FsStore manages items under the given base filesystem.

    Note that FsStore has no consistency checks; it assumes that no direct modification
    of the files in its directory has occurred.

    """

    fs: FS
    _item_type: t.Type[StoreItem]

    @abstractmethod
    def __init__(self, base_path: PathType, item_type: t.Type[StoreItem]):
        self._item_type = item_type
        self.fs = fs.open_fs(str(base_path))

    def list(self, tag: t.Optional[t.Union[Tag, str]] = None) -> t.List[Tag]:
        if not tag:
            return sorted([ver for _d in self.fs.listdir("/") for ver in self.list(_d)])

        _tag = Tag.from_taglike(tag)
        if _tag.version is None:
            tags = [
                Tag(_tag.name, f.name) for f in self.fs.scandir(_tag.name) if f.is_dir
            ]
            return sorted(tags)
        else:
            return [_tag] if self.fs.isdir(_tag.path()) else []

    def _get_item(self, tag: Tag) -> Item:
        """
        Creates a new instance of Item that represents the item with tag `tag`.
        """
        return self._item_type.from_fs(tag, self.fs.opendir(tag.path()))  # type: ignore

    def get(self, tag: t.Union[Tag, str]) -> Item:
        """
        store.get("my_bento")
        store.get("my_bento:v1.0.0")
        store.get(Tag("my_bento", "latest"))
        """
        _tag = Tag.from_taglike(tag)
        if _tag.version is None or _tag.version == "latest":
            _tag.version = self.fs.readtext(_tag.latest_path())
        path = _tag.path()
        if not self.fs.exists(path):
            raise FileNotFoundError(
                f"Item '{tag}' is not found in BentoML store {self.fs}."
            )
        return self._get_item(_tag)

    @contextmanager
    def register(self, tag: t.Union[str, Tag]):
        _tag = Tag.from_taglike(tag)

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
            with self.fs.open(_tag.latest_path(), "w") as latest_file:
                latest_file.write(_tag.version)

    def delete(self, tag: t.Union[str, Tag]) -> None:
        _tag = Tag.from_taglike(tag)

        self.fs.removetree(_tag.path())
        if self.fs.isdir(_tag.name):
            versions = self.list(_tag.name)
            if len(versions) == 0:
                # if we've removed all versions, remove the directory
                self.fs.removetree(_tag.name)
            else:
                new_latest = sorted(
                    versions, key=lambda tag: self._get_item(tag).creation_time()
                )[0]
                # otherwise, update the latest version
                self.fs.writetext(_tag.latest_path(), new_latest.name)
