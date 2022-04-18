import os
import typing as t
import datetime
from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager

import fs
import fs.errors
from fs.base import FS

from .tag import Tag
from .types import PathType
from .exportable import Exportable
from ..exceptions import NotFound
from ..exceptions import BentoMLException

T = t.TypeVar("T")


class StoreItem(Exportable):
    @property
    @abstractmethod
    def tag(self) -> Tag:
        raise NotImplementedError

    @property
    def _fs(self) -> FS:
        raise NotImplementedError

    @classmethod
    def get_typename(cls) -> str:
        return cls.__name__

    @property
    def _export_name(self) -> str:
        return f"{self.tag.name}-{self.tag.version}"

    @classmethod
    @abstractmethod
    def from_fs(cls: t.Type[T], item_fs: FS) -> T:
        raise NotImplementedError

    @property
    @abstractmethod
    def creation_time(self) -> datetime.datetime:
        raise NotImplementedError

    def __repr__(self):
        return f'{self.get_typename()}(tag="{self.tag}")'


Item = t.TypeVar("Item", bound=StoreItem)


class Store(ABC, t.Generic[Item]):
    """An FsStore manages items under the given base filesystem.

    Note that FsStore has no consistency checks; it assumes that no direct modification
    of the files in its directory has occurred.

    """

    _fs: FS
    _item_type: t.Type[Item]

    @abstractmethod
    def __init__(self, base_path: t.Union[PathType, FS], item_type: t.Type[Item]):
        self._item_type = item_type
        if isinstance(base_path, os.PathLike):
            base_path = base_path.__fspath__()
        self._fs = fs.open_fs(base_path)

    def list(self, tag: t.Optional[t.Union[Tag, str]] = None) -> t.List[Item]:
        if not tag:
            return [
                ver
                for _d in sorted(self._fs.listdir("/"))
                if self._fs.isdir(_d)
                for ver in self.list(_d)
            ]

        _tag = Tag.from_taglike(tag)
        if _tag.version is None:
            if not self._fs.isdir(_tag.name):
                raise NotFound(
                    f"no {self._item_type.get_typename()}s with name '{_tag.name}' found"
                )

            tags = sorted(
                [
                    Tag(_tag.name, f.name)
                    for f in self._fs.scandir(_tag.name)
                    if f.is_dir
                ]
            )
            return [self._get_item(t) for t in tags]
        else:
            return [self._get_item(_tag)] if self._fs.isdir(_tag.path()) else []

    def _get_item(self, tag: Tag) -> Item:
        """
        Creates a new instance of Item that represents the item with tag `tag`.
        """
        return self._item_type.from_fs(self._fs.opendir(tag.path()))

    def _recreate_latest(self, tag: Tag):
        try:
            items = self.list(tag.name)
        except NotFound:
            raise NotFound(
                f"no {self._item_type.get_typename()}s with name '{tag.name}' exist in BentoML store {self._fs}"
            )

        if len(items) == 0:
            raise NotFound(
                f"no {self._item_type.get_typename()}s with name '{tag.name}' exist in BentoML store {self._fs}"
            )

        items.sort(reverse=True, key=lambda item: item.creation_time)
        tag.version = items[0].tag.version

        with self._fs.open(tag.latest_path(), "w") as latest_file:
            latest_file.write(tag.version)

    def get(self, tag: t.Union[Tag, str]) -> Item:
        """
        store.get("my_bento")
        store.get("my_bento:v1.0.0")
        store.get(Tag("my_bento", "latest"))
        """
        _tag = Tag.from_taglike(tag)
        if _tag.version is None or _tag.version == "latest":
            try:
                _tag.version = self._fs.readtext(_tag.latest_path())

                if not self._fs.exists(_tag.path()):
                    self._recreate_latest(_tag)
            except fs.errors.ResourceNotFound:
                self._recreate_latest(_tag)

        path = _tag.path()
        if self._fs.exists(path):
            return self._get_item(_tag)

        matches = self._fs.glob(f"{path}*/")
        counts = matches.count().directories
        if counts == 0:
            raise NotFound(
                f"{self._item_type.get_typename()} '{tag}' is not found in BentoML store {self._fs}"
            )
        elif counts == 1:
            match = next(iter(matches))
            return self._get_item(Tag(_tag.name, match.info.name))
        else:
            vers: t.List[str] = []
            for match in matches:
                vers += match.info.name
            raise BentoMLException(
                f"multiple versions matched by {_tag.version}: {vers}"
            )

    @contextmanager
    def register(self, tag: t.Union[str, Tag]):
        _tag = Tag.from_taglike(tag)

        item_path = _tag.path()
        if self._fs.exists(item_path):
            raise BentoMLException(
                f"Item '{_tag}' already exists in the store {self._fs}"
            )
        self._fs.makedirs(item_path)
        try:
            yield self._fs.getsyspath(item_path)
        finally:
            # item generation is most likely successful, link latest path
            if (
                not self._fs.exists(_tag.latest_path())
                or self.get(_tag).creation_time > self.get(_tag.name).creation_time
            ):
                with self._fs.open(_tag.latest_path(), "w") as latest_file:
                    latest_file.write(_tag.version)

    def delete(self, tag: t.Union[str, Tag]) -> None:
        _tag = Tag.from_taglike(tag)

        if not self._fs.exists(_tag.path()):
            raise NotFound(f"{self._item_type.get_typename()} '{tag}' not found")

        self._fs.removetree(_tag.path())
        if self._fs.isdir(_tag.name):
            versions = self.list(_tag.name)
            if len(versions) == 0:
                # if we've removed all versions, remove the directory
                self._fs.removetree(_tag.name)
            else:
                new_latest = sorted(versions, key=lambda x: x.creation_time)[-1]
                # otherwise, update the latest version
                assert new_latest.tag.version is not None
                self._fs.writetext(_tag.latest_path(), new_latest.tag.version)
