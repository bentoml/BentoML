import typing as t
from datetime import datetime

import fs
import attr
import pytest
from fs.base import FS

from bentoml.exceptions import NotFound
from bentoml.exceptions import BentoMLException
from bentoml._internal.store import Store
from bentoml._internal.store import StoreItem
from bentoml._internal.types import Tag


@attr.define(repr=False)
class DummyItem(StoreItem):
    tag: Tag
    fs: FS
    creation_time: datetime = attr.field(factory=datetime.now)
    store: "DummyStore" = None

    @staticmethod
    def create(tag: t.Union[str, Tag]):
        with DummyItem.store.register(tag) as path:
            fs.open_fs(path).writetext("tag", str(tag))

    @classmethod
    def from_fs(cls, dummy_fs):
        return DummyItem(Tag.from_str(dummy_fs.readtext("tag")), dummy_fs)


class DummyStore(Store[DummyItem]):
    def __init__(self, base_path):
        super().__init__(base_path, DummyItem)


def test_store(tmpdir):
    store = DummyStore(tmpdir)

    DummyItem.store = store
    DummyItem.create("test:version1")
    DummyItem.create("test:otherprefix")
    DummyItem.create(Tag("test", "version2"))
    DummyItem.create("test1:version1")
    with pytest.raises(BentoMLException):
        DummyItem.create("test:version2")

    item = store.get("test:version1")
    assert item.tag == Tag("test", "version1")
    item = store.get("test:oth")
    assert item.tag == Tag("test", "otherprefix")
    latest = store.get("test:latest")
    assert latest.tag == Tag("test", "version2")

    with pytest.raises(BentoMLException):
        store.get("test:ver")

    with pytest.raises(NotFound):
        store.get("nonexistent:latest")
    with pytest.raises(NotFound):
        store.get("test:version3")

    vers = store.list("test")
    assert set([ver.tag for ver in vers]) == {
        Tag("test", "version1"),
        Tag("test", "version2"),
        Tag("test", "otherprefix"),
    }

    vers = store.list()
    assert set([ver.tag for ver in vers]) == {
        Tag("test", "version1"),
        Tag("test", "version2"),
        Tag("test", "otherprefix"),
        Tag("test1", "version1"),
    }
