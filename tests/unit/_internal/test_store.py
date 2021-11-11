from datetime import datetime

import attr
import pytest
from fs.base import FS

from bentoml._internal.store import Store, StoreItem
from bentoml._internal.types import Tag
from bentoml.exceptions import BentoMLException, NotFound


@attr.define(repr=False)
class DummyItem(StoreItem):
    tag: Tag
    fs: FS
    creation_time: datetime = attr.field(factory=datetime.now)

    @classmethod
    def from_fs(cls, tag, fs):
        return DummyItem(tag, fs)


class DummyStore(Store[DummyItem]):
    def __init__(self, base_path):
        super().__init__(base_path, DummyItem)


def test_store(tmpdir):
    store = DummyStore(tmpdir)

    with store.register("test:version1"):
        pass
    with store.register("test:otherprefix"):
        pass
    with store.register("test:version2"):
        pass
    with store.register("test1:version1"):
        pass
    with pytest.raises(BentoMLException):
        with store.register("test:version2"):
            pass

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
