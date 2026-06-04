from __future__ import annotations

import io
import os
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

import bentoml
from bentoml import Tag
from bentoml._internal.bento import Bento
from bentoml._internal.bento.bento import BentoInfo
from bentoml._internal.exportable import Exportable
from bentoml._internal.exportable import _normalize_member_path
from bentoml._internal.models import Model
from bentoml._internal.models import ModelStore
from bentoml._internal.types import PathType
from bentoml.testing.pytest import TEST_MODEL_CONTEXT


class DummyExportable(Exportable):
    def __init__(self, path: PathType):
        self._path = Path(path)

    @staticmethod
    def _export_ext() -> str:
        return "dummy"

    @property
    def _export_name(self) -> str:
        return "dummy"

    @classmethod
    def from_path(cls, path: PathType) -> DummyExportable:
        if not Path(path).joinpath("marker.txt").is_file():
            raise RuntimeError("missing marker")
        return cls(path)


class FailingExportable(DummyExportable):
    @classmethod
    def from_path(cls, path: PathType) -> FailingExportable:
        raise RuntimeError("from_path failed")


def _add_tar_file(tar: tarfile.TarFile, name: str, data: bytes = b"data") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _add_tar_dir(tar: tarfile.TarFile, name: str) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    tar.addfile(info)


def _add_tar_symlink(tar: tarfile.TarFile, name: str, target: str) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    tar.addfile(info)


def _add_tar_hardlink(tar: tarfile.TarFile, name: str, target: str) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE
    info.linkname = target
    tar.addfile(info)


def _tar_fs(tmp_path: Path, builder) -> object:
    from fsspec.implementations.tar import TarFileSystem

    path = tmp_path / "archive.tar"
    with tarfile.open(path, "w") as tar:
        builder(tar)
    return TarFileSystem(str(path), compression=None)


def _zip_fs(tmp_path: Path, builder) -> object:
    from fsspec.implementations.zip import ZipFileSystem

    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as zf:
        builder(zf)
    return ZipFileSystem(str(path))


def _assert_unsafe(exc: pytest.ExceptionInfo[ValueError]) -> None:
    assert "Unsafe archive member path" in str(exc.value)


@pytest.mark.parametrize("archive_kind", ["tar", "zip"])
def test_from_fs_imports_safe_nested_archive(tmp_path: Path, archive_kind: str) -> None:
    if archive_kind == "tar":
        fs = _tar_fs(
            tmp_path,
            lambda tar: (
                _add_tar_dir(tar, "./"),
                _add_tar_dir(tar, "nested"),
                _add_tar_file(tar, "marker.txt", b"ok"),
                _add_tar_file(tar, "nested/file.txt", b"nested"),
            ),
        )
    else:
        fs = _zip_fs(
            tmp_path,
            lambda zf: (
                zf.writestr("nested/", b""),
                zf.writestr("marker.txt", b"ok"),
                zf.writestr("nested/file.txt", b"nested"),
            ),
        )

    imported = DummyExportable._from_fs(fs)  # type: ignore[arg-type]

    assert imported._path.joinpath("marker.txt").read_text() == "ok"
    assert imported._path.joinpath("nested/file.txt").read_text() == "nested"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="os.symlink is unavailable")
def test_from_fs_preserves_safe_tar_symlink(tmp_path: Path) -> None:
    fs = _tar_fs(
        tmp_path,
        lambda tar: (
            _add_tar_dir(tar, "nested"),
            _add_tar_file(tar, "marker.txt", b"ok"),
            _add_tar_symlink(tar, "marker-link.txt", "marker.txt"),
            _add_tar_symlink(tar, "nested/marker-link.txt", "../marker.txt"),
        ),
    )

    imported = DummyExportable._from_fs(fs)  # type: ignore[arg-type]
    link = imported._path / "marker-link.txt"
    nested_link = imported._path / "nested" / "marker-link.txt"

    assert link.is_symlink()
    assert os.readlink(link) == "marker.txt"
    assert link.read_text() == "ok"
    assert nested_link.is_symlink()
    assert os.readlink(nested_link) == "../marker.txt"
    assert nested_link.read_text() == "ok"


@pytest.mark.parametrize(
    "member_name",
    [
        "../outside.txt",
        "safe/../evil.txt",
        "/absolute.txt",
        "C:relative.txt",
        "C:\\absolute.txt",
        "\\rooted.txt",
        "\\\\server\\share\\evil.txt",
        "safe\\..\\evil.txt",
    ],
)
@pytest.mark.parametrize("archive_kind", ["tar", "zip"])
def test_from_fs_rejects_unsafe_member_paths(
    tmp_path: Path,
    archive_kind: str,
    member_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bentoml._internal.exportable as exportable_mod

    temp_dir = tmp_path / "import-temp"
    outside = tmp_path / "outside.txt"
    monkeypatch.setattr(
        exportable_mod.tempfile,
        "mkdtemp",
        lambda *args, **kwargs: str(temp_dir),
    )
    if archive_kind == "tar":
        fs = _tar_fs(
            tmp_path,
            lambda tar: (
                _add_tar_file(tar, "marker.txt", b"ok"),
                _add_tar_file(tar, member_name, b"pwned"),
            ),
        )
    else:
        fs = _zip_fs(
            tmp_path,
            lambda zf: (
                zf.writestr("marker.txt", b"ok"),
                zf.writestr(member_name, b"pwned"),
            ),
        )

    with pytest.raises(ValueError) as exc:
        DummyExportable._from_fs(fs)  # type: ignore[arg-type]

    _assert_unsafe(exc)
    assert not outside.exists()


@pytest.mark.parametrize(
    "target",
    [
        "../outside.txt",
        "/absolute.txt",
        "C:relative.txt",
        "C:\\absolute.txt",
        "\\rooted.txt",
        "\\\\server\\share\\evil.txt",
        "safe\\..\\evil.txt",
    ],
)
def test_from_fs_rejects_unsafe_tar_symlink_targets(
    tmp_path: Path, target: str
) -> None:
    fs = _tar_fs(
        tmp_path,
        lambda tar: (
            _add_tar_file(tar, "marker.txt", b"ok"),
            _add_tar_symlink(tar, "marker-link.txt", target),
        ),
    )

    with pytest.raises(ValueError) as exc:
        DummyExportable._from_fs(fs)  # type: ignore[arg-type]

    _assert_unsafe(exc)


def test_path_validator_rejects_nul_bytes() -> None:
    with pytest.raises(ValueError) as exc:
        _normalize_member_path("nul\x00evil.txt")

    _assert_unsafe(exc)


def test_from_fs_rejects_zip_symlink_entries(tmp_path: Path) -> None:
    def build(zf: zipfile.ZipFile) -> None:
        zf.writestr("marker.txt", b"ok")
        info = zipfile.ZipInfo("marker-link.txt")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, b"marker.txt")

    fs = _zip_fs(tmp_path, build)

    with pytest.raises(ValueError) as exc:
        DummyExportable._from_fs(fs)  # type: ignore[arg-type]

    _assert_unsafe(exc)


def test_from_fs_rejects_tar_hardlinks(tmp_path: Path) -> None:
    fs = _tar_fs(
        tmp_path,
        lambda tar: (
            _add_tar_file(tar, "marker.txt", b"ok"),
            _add_tar_hardlink(tar, "marker-hardlink.txt", "marker.txt"),
        ),
    )

    with pytest.raises(ValueError) as exc:
        DummyExportable._from_fs(fs)  # type: ignore[arg-type]

    _assert_unsafe(exc)


def test_from_fs_rejects_tar_special_entries(tmp_path: Path) -> None:
    def build(tar: tarfile.TarFile) -> None:
        _add_tar_file(tar, "marker.txt", b"ok")
        info = tarfile.TarInfo("fifo")
        info.type = tarfile.FIFOTYPE
        tar.addfile(info)

    fs = _tar_fs(tmp_path, build)

    with pytest.raises(ValueError) as exc:
        DummyExportable._from_fs(fs)  # type: ignore[arg-type]

    _assert_unsafe(exc)


def test_from_fs_rejects_duplicate_file_destinations(tmp_path: Path) -> None:
    fs = _tar_fs(
        tmp_path,
        lambda tar: (
            _add_tar_file(tar, "marker.txt", b"first"),
            _add_tar_file(tar, "./marker.txt", b"second"),
        ),
    )

    with pytest.raises(ValueError) as exc:
        DummyExportable._from_fs(fs)  # type: ignore[arg-type]

    _assert_unsafe(exc)


def test_from_fs_rejects_duplicate_file_symlink_destinations(tmp_path: Path) -> None:
    fs = _tar_fs(
        tmp_path,
        lambda tar: (
            _add_tar_file(tar, "marker.txt", b"ok"),
            _add_tar_symlink(tar, "./marker.txt", "marker.txt"),
        ),
    )

    with pytest.raises(ValueError) as exc:
        DummyExportable._from_fs(fs)  # type: ignore[arg-type]

    _assert_unsafe(exc)


def test_from_fs_allows_repeated_directory_entries(tmp_path: Path) -> None:
    fs = _tar_fs(
        tmp_path,
        lambda tar: (
            _add_tar_dir(tar, "nested"),
            _add_tar_dir(tar, "./nested"),
            _add_tar_file(tar, "marker.txt", b"ok"),
            _add_tar_file(tar, "nested/file.txt", b"nested"),
        ),
    )

    imported = DummyExportable._from_fs(fs)  # type: ignore[arg-type]

    assert imported._path.joinpath("nested/file.txt").read_text() == "nested"


def test_from_fs_cleans_up_after_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bentoml._internal.exportable as exportable_mod

    temp_dir = tmp_path / "import-temp"
    monkeypatch.setattr(
        exportable_mod.tempfile,
        "mkdtemp",
        lambda *args, **kwargs: str(temp_dir),
    )
    fs = _tar_fs(
        tmp_path,
        lambda tar: (
            _add_tar_file(tar, "marker.txt", b"ok"),
            _add_tar_file(tar, "../outside.txt", b"pwned"),
        ),
    )

    with pytest.raises(ValueError):
        DummyExportable._from_fs(fs)  # type: ignore[arg-type]

    assert not temp_dir.exists()


def test_from_fs_cleans_up_after_from_path_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bentoml._internal.exportable as exportable_mod

    temp_dir = tmp_path / "import-temp"
    monkeypatch.setattr(
        exportable_mod.tempfile,
        "mkdtemp",
        lambda *args, **kwargs: str(temp_dir),
    )
    fs = _tar_fs(tmp_path, lambda tar: _add_tar_file(tar, "marker.txt", b"ok"))

    with pytest.raises(RuntimeError, match="from_path failed"):
        FailingExportable._from_fs(fs)  # type: ignore[arg-type]

    assert not temp_dir.exists()


def test_from_fs_imports_fsspec_directory_source(tmp_path: Path) -> None:
    import fsspec
    from fsspec.implementations.dirfs import DirFileSystem

    memory_fs = fsspec.filesystem("memory")
    root = f"/test-exportable-{id(tmp_path)}"
    memory_fs.makedirs(f"{root}/nested", exist_ok=True)
    memory_fs.pipe(f"{root}/marker.txt", b"ok")
    memory_fs.pipe(f"{root}/nested/file.txt", b"nested")
    fs = DirFileSystem(root, fs=memory_fs)

    imported = DummyExportable._from_fs(fs)

    assert imported._path.joinpath("marker.txt").read_text() == "ok"
    assert imported._path.joinpath("nested/file.txt").read_text() == "nested"


def test_import_model_rejects_otherwise_valid_archive_with_traversal(
    tmp_path: Path,
) -> None:
    model = Model.create(
        "security-test-model:v1",
        module=__name__,
        api_version="v1",
        signatures={},
        context=TEST_MODEL_CONTEXT,
    )
    model.flush()
    export_path = model.export(str(tmp_path / "model.tar"), "tar")
    with tarfile.open(export_path, "a") as tar:
        _add_tar_file(tar, "../outside-model.txt", b"pwned")

    with pytest.raises(ValueError) as exc:
        bentoml.models.import_model(
            export_path,
            "tar",
            _model_store=ModelStore(tmp_path / "models"),
        )

    _assert_unsafe(exc)
    assert not tmp_path.joinpath("outside-model.txt").exists()


def test_import_bento_rejects_otherwise_valid_archive_with_traversal(
    tmp_path: Path,
) -> None:
    from bentoml._internal.bento import BentoStore

    bento_path = tmp_path / "bento-source"
    bento_path.mkdir()
    info = BentoInfo(tag=Tag("security-test-bento", "v1"), service="service.py:svc")
    bento = Bento(info.tag, bento_path, info)
    bento.flush_info()
    export_path = bento.export(str(tmp_path / "bento.tar"), "tar")
    with tarfile.open(export_path, "a") as tar:
        _add_tar_file(tar, "../outside-bento.txt", b"pwned")

    with pytest.raises(ValueError) as exc:
        bentoml.import_bento(
            export_path,
            "tar",
            _bento_store=BentoStore(tmp_path / "bentos"),
        )

    _assert_unsafe(exc)
    assert not tmp_path.joinpath("outside-bento.txt").exists()
