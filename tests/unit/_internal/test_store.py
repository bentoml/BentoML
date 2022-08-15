import os
import sys
import time
import typing as t
from typing import TYPE_CHECKING
from datetime import datetime

import fs
import attr
import pytest

from bentoml import Tag
from bentoml.exceptions import NotFound
from bentoml.exceptions import BentoMLException
from bentoml._internal.store import Store
from bentoml._internal.store import StoreItem

if sys.version_info < (3, 7):
    from backports.datetime_fromisoformat import MonkeyPatch

    MonkeyPatch.patch_fromisoformat()

if TYPE_CHECKING:
    from pathlib import Path

    from fs.base import FS

    from bentoml._internal.types import PathType


@attr.define(repr=False)
class DummyItem(StoreItem):
    _tag: Tag
    fs: "FS"
    _creation_time: datetime
    store: "DummyStore" = attr.field(init=False)

    @staticmethod
    def _export_ext() -> str:
        return "bentodummy"

    @property
    def tag(self) -> Tag:
        return self._tag

    @property
    def creation_time(self) -> datetime:
        return self._creation_time

    @staticmethod
    def create(tag: t.Union[str, Tag], creation_time: t.Optional[datetime] = None):
        creation_time = datetime.now() if creation_time is None else creation_time
        with DummyItem.store.register(tag) as path:
            dummy_fs = fs.open_fs(path)
            dummy_fs.writetext("tag", str(tag))
            dummy_fs.writetext("ctime", creation_time.isoformat())

    @classmethod
    def from_fs(cls, item_fs: "FS") -> "DummyItem":
        return DummyItem(
            Tag.from_str(item_fs.readtext("tag")),
            item_fs,
            datetime.fromisoformat(item_fs.readtext("ctime")),
        )


class DummyStore(Store[DummyItem]):
    def __init__(self, base_path: "t.Union[PathType, FS]"):
        super().__init__(base_path, DummyItem)


def test_store(tmpdir: "Path"):
    store = DummyStore(tmpdir)

    open(os.path.join(tmpdir, ".DS_store"), "a", encoding="utf-8")

    DummyItem.store = store
    oldtime = datetime.now()
    DummyItem.create("test:version1")
    time.sleep(1)
    DummyItem.create("test:otherprefix")
    time.sleep(1)
    DummyItem.create(Tag("test", "version2"))
    time.sleep(1)
    DummyItem.create("test:version3", creation_time=oldtime)
    time.sleep(1)
    DummyItem.create("test1:version1")
    with pytest.raises(BentoMLException):
        DummyItem.create("test:version2")

    item = store.get("test:version1")
    assert item.tag == Tag("test", "version1")
    item = store.get("test:oth")
    assert item.tag == Tag("test", "otherprefix")
    latest = store.get("test:latest")
    assert latest.tag == Tag("test", "version2")
    latest = store.get("test")
    assert latest.tag == Tag("test", "version2")

    with pytest.raises(BentoMLException):
        store.get("test:ver")

    with pytest.raises(NotFound):
        store.get("nonexistent:latest")
    with pytest.raises(NotFound):
        store.get("test:version4")

    vers = store.list()
    assert set([ver.tag for ver in vers]) == {
        Tag("test", "version1"),
        Tag("test", "version2"),
        Tag("test", "version3"),
        Tag("test", "otherprefix"),
        Tag("test1", "version1"),
    }

    vers = store.list("test")
    assert set([ver.tag for ver in vers]) == {
        Tag("test", "version1"),
        Tag("test", "version2"),
        Tag("test", "version3"),
        Tag("test", "otherprefix"),
    }

    vers = store.list("test:version1")
    assert set([ver.tag for ver in vers]) == {Tag("test", "version1")}

    assert store.list("nonexistent:latest") == []
    assert store.list("test:version4") == []

    store.delete("test:version2")
    latest = store.get("test")
    assert latest.tag == Tag("test", "otherprefix")

    with pytest.raises(NotFound):
        store.delete("test:version4")

    store.delete("test1:version1")

    with pytest.raises(NotFound):
        store.get("test1")

    with pytest.raises(NotFound):
        store.list("test1")

    store.delete("test")

    with pytest.raises(NotFound):
        store.list("test")

    assert store.list() == []
