from __future__ import annotations

import datetime
import shutil
import typing as t
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path

from ..exceptions import BentoMLException
from ..exceptions import NotFound
from .exportable import Exportable
from .tag import Tag
from .types import PathType
from .utils.filesystem import calc_dir_size

T = t.TypeVar("T")


class StoreItem(Exportable):
    @property
    @abstractmethod
    def tag(self) -> Tag:
        raise NotImplementedError

    @classmethod
    def get_typename(cls) -> str:
        return cls.__name__

    @property
    def _export_name(self) -> str:
        return f"{self.tag.name}-{self.tag.version}"

    @property
    @abstractmethod
    def creation_time(self) -> datetime.datetime:
        raise NotImplementedError

    @property
    def path(self) -> str:
        return str(self._path)

    def path_of(self, item: str) -> str:
        return str(self._path / item.lstrip("/"))

    @property
    def file_size(self) -> int:
        return calc_dir_size(self._path)

    def __repr__(self):
        return f'{self.get_typename()}(tag="{self.tag}")'


Item = t.TypeVar("Item", bound=StoreItem)


class Store(t.Generic[Item]):
    """A Store manages items under the given base filesystem.

    Note that Store has no consistency checks; it assumes that no direct modification
    of the files in its directory has occurred.

    """

    _item_type: t.Type[Item]

    def __init__(self, base_path: PathType):
        self._path = Path(base_path)

    def list(self, tag: Tag | str | None = None) -> t.List[Item]:
        if not tag:
            # Return all items in the store
            return [
                ver
                for _d in sorted(self._path.iterdir())
                if _d.is_dir()
                for ver in self.list(_d.name)
            ]

        _tag = Tag.from_taglike(tag)
        if _tag.version is None:
            item_dir = self._path / _tag.name
            if not item_dir.is_dir():
                raise NotFound(
                    f"no {self._item_type.get_typename()}s with name '{_tag.name}' found"
                )

            tags = sorted(
                Tag(_tag.name, f.name) for f in item_dir.iterdir() if f.is_dir()
            )
            return [self._get_item(t) for t in tags]
        else:
            return [self._get_item(_tag)] if (self._path / _tag.path()).is_dir() else []

    def _get_item(self, tag: Tag) -> Item:
        """
        Creates a new instance of Item that represents the item with tag `tag`.
        """
        return self._item_type.from_path(self._path / tag.path())

    def _recreate_latest(self, tag: Tag):
        try:
            items = self.list(tag.name)
        except NotFound:
            raise NotFound(
                f"no {self._item_type.get_typename()}s with name '{tag.name}' exist in BentoML store {self._path}"
            )

        if len(items) == 0:
            raise NotFound(
                f"no {self._item_type.get_typename()}s with name '{tag.name}' exist in BentoML store {self._path}"
            )

        items.sort(reverse=True, key=lambda item: item.creation_time)
        latest_version = t.cast(str, items[0].tag.version)

        with (self._path / tag.latest_path()).open("w") as latest_file:
            latest_file.write(latest_version)
        tag.version = latest_version

    def get(self, tag: Tag | str) -> Item:
        """
        store.get("my_bento")
        store.get("my_bento:v1.0.0")
        store.get(Tag("my_bento", "latest"))
        """
        _tag = Tag.from_taglike(tag)
        if _tag.version is None or _tag.version == "latest":
            try:
                _tag.version = (self._path / _tag.latest_path()).read_text().strip()

                if not (self._path / _tag.path()).exists():
                    self._recreate_latest(_tag)
            except OSError:
                self._recreate_latest(_tag)

        path = _tag.path()
        if (self._path / path).exists():
            return self._get_item(_tag)

        matches = list(p for p in self._path.glob(f"{path}*/") if p.is_dir())
        if len(matches) == 1:
            return self._get_item(Tag(_tag.name, matches[0].name))
        elif len(matches) > 1:
            vers = [p.name for p in matches]
            raise BentoMLException(
                f"Multiple {_tag.name} versions found: {', '.join(vers)}"
            )
        # No match found
        cmd = (
            "bentoml models pull"
            if self._item_type.get_typename() == "Model"
            else "bentoml pull"
        )
        raise NotFound(
            f"{self._item_type.get_typename()} '{tag}' is not found in BentoML store {self._path}, "
            f"you may need to run `{cmd}` first"
        )

    @contextmanager
    def register(self, tag: str | Tag) -> t.Generator[str, None, None]:
        _tag = Tag.from_taglike(tag)

        item_path = _tag.path()
        if (self._path / item_path).exists():
            raise BentoMLException(
                f"Item '{_tag}' already exists in the store {self._path}"
            )
        (self._path / item_path).mkdir(parents=True, exist_ok=True)
        try:
            yield str(self._path / item_path)
        except Exception:
            shutil.rmtree(self._path / item_path)
            raise
        # item generation is most likely successful, link latest path
        if (
            not (latest_file := self._path / _tag.latest_path()).exists()
            or self.get(_tag).creation_time >= self.get(_tag.name).creation_time
        ):
            with latest_file.open("w") as f:
                f.write(_tag.version)

    def delete(self, tag: str | Tag) -> None:
        _tag = Tag.from_taglike(tag)

        if _tag.version == "latest":
            try:
                _tag.version = (self._path / _tag.latest_path()).read_text().strip()
            except OSError:
                # if latest path doesn't exist, we don't need to delete anything
                return

        tag_path = self._path / _tag.path()
        if not tag_path.exists():
            raise NotFound(f"{self._item_type.get_typename()} '{tag}' not found")

        shutil.rmtree(tag_path)
        if (self._path / _tag.name).is_dir():
            versions = self.list(_tag.name)
            if len(versions) == 0:
                # if we've removed all versions, remove the directory
                shutil.rmtree(self._path / _tag.name)
            else:
                new_latest = sorted(versions, key=lambda x: x.creation_time)[-1]
                # otherwise, update the latest version
                assert new_latest.tag.version is not None
                (self._path / _tag.latest_path()).write_text(new_latest.tag.version)
