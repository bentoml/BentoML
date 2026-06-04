from __future__ import annotations

import logging
import os
import posixpath
import re
import shutil
import stat
import tempfile
import typing as t
import urllib.parse
import weakref
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import fsspec
from fsspec.registry import known_implementations

from .types import PathType
from .utils.filesystem import safe_remove_dir

T = t.TypeVar("T", bound="Exportable")

_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:")
_UNSAFE_MEMBER_MESSAGE = "Unsafe archive member path"


@dataclass(frozen=True)
class _ImportEntry:
    source: t.Any
    path: str
    kind: t.Literal["directory", "file", "symlink"]
    linkname: str | None = None


def _raise_unsafe_member(path: str, reason: str) -> t.NoReturn:
    raise ValueError(f"{_UNSAFE_MEMBER_MESSAGE}: {path!r} ({reason})")


def _normalize_member_path(
    path: str, *, description: str = "member", allow_root: bool = True
) -> str | None:
    if "\x00" in path:
        _raise_unsafe_member(path, f"{description} contains NUL byte")
    if path.startswith(("/", "\\")):
        _raise_unsafe_member(path, f"{description} is absolute")
    if _WINDOWS_DRIVE_PATH.match(path):
        _raise_unsafe_member(path, f"{description} is drive-qualified")

    path = path.replace("\\", "/")
    if path.startswith("/") or path.startswith("//"):
        _raise_unsafe_member(path, f"{description} is absolute")
    if _WINDOWS_DRIVE_PATH.match(path):
        _raise_unsafe_member(path, f"{description} is drive-qualified")

    parts = [part for part in path.split("/") if part not in ("", ".")]
    if not parts:
        if allow_root:
            return None
        _raise_unsafe_member(path, f"{description} is empty")
    if any(part == ".." for part in parts):
        _raise_unsafe_member(path, f"{description} contains path traversal")
    return posixpath.normpath(posixpath.join(*parts))


def _normalize_fsspec_path(path: str) -> str | None:
    return _normalize_member_path(path.lstrip("/"), description="filesystem path")


def _validate_symlink_target(linkname: str, destination: str, root: str) -> None:
    if "\x00" in linkname:
        _raise_unsafe_member(linkname, "symlink target contains NUL byte")
    if not linkname:
        _raise_unsafe_member(linkname, "symlink target is empty")
    if linkname.startswith("/"):
        _raise_unsafe_member(linkname, "symlink target is absolute")
    if "\\" in linkname:
        _raise_unsafe_member(linkname, "symlink target contains backslash")
    if _WINDOWS_DRIVE_PATH.match(linkname):
        _raise_unsafe_member(linkname, "symlink target is drive-qualified")

    resolved_target = os.path.join(os.path.dirname(destination), linkname)
    _ensure_within_directory(root, resolved_target)


def _realpath(path: str | os.PathLike[str]) -> str:
    return os.path.realpath(os.fspath(path))


def _ensure_within_directory(root: str, path: str | os.PathLike[str]) -> None:
    real_path = _realpath(path)
    if os.path.commonpath([root, real_path]) != root:
        _raise_unsafe_member(os.fspath(path), "destination escapes import directory")


def _validate_import_entries(entries: list[_ImportEntry], temp_dir: str) -> None:
    root = _realpath(temp_dir)
    seen: dict[str, str] = {}
    symlink_paths: set[str] = set()

    for entry in entries:
        previous_kind = seen.get(entry.path)
        if previous_kind is not None:
            if previous_kind == entry.kind == "directory":
                continue
            _raise_unsafe_member(entry.path, "duplicate destination path")
        seen[entry.path] = entry.kind

        destination = os.path.join(root, entry.path)
        _ensure_within_directory(root, destination)

        if entry.kind == "symlink":
            assert entry.linkname is not None
            symlink_paths.add(entry.path)
            _validate_symlink_target(entry.linkname, destination, root)

    for symlink_path in symlink_paths:
        prefix = f"{symlink_path}/"
        for entry in entries:
            if entry.path.startswith(prefix):
                _raise_unsafe_member(
                    entry.path, f"entry is nested under symlink {symlink_path!r}"
                )


def _copy_regular_file(source: t.IO[bytes], destination: str, temp_dir: str) -> None:
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    _ensure_within_directory(temp_dir, parent)
    _ensure_within_directory(temp_dir, destination)
    with open(destination, "wb") as dest:
        shutil.copyfileobj(source, dest)


def _copy_tar_fs(fs: fsspec.AbstractFileSystem, temp_dir: str) -> None:
    import tarfile

    tar = getattr(fs, "tar", None)
    if not isinstance(tar, tarfile.TarFile):
        raise ValueError("Unsupported tar filesystem object")

    entries: list[_ImportEntry] = []
    for member in tar.getmembers():
        path = _normalize_member_path(member.name)
        if path is None:
            continue
        if member.isdir():
            entries.append(_ImportEntry(member, path, "directory"))
        elif member.issym():
            entries.append(_ImportEntry(member, path, "symlink", member.linkname))
        elif member.islnk():
            # shutil.make_archive can emit hardlinks from hardlinked source trees.
            # Safe support should materialize them as copies after validating the
            # link target is an already validated regular file member.
            raise ValueError(f"{_UNSAFE_MEMBER_MESSAGE}: hardlinks are unsupported")
        elif member.isfile():
            entries.append(_ImportEntry(member, path, "file"))
        else:
            raise ValueError(
                f"{_UNSAFE_MEMBER_MESSAGE}: unsupported tar member {member.name!r}"
            )

    _validate_import_entries(entries, temp_dir)

    root = _realpath(temp_dir)
    for entry in entries:
        destination = os.path.join(root, entry.path)
        if entry.kind == "directory":
            os.makedirs(destination, exist_ok=True)
            _ensure_within_directory(root, destination)

    for entry in entries:
        if entry.kind != "file":
            continue
        source = tar.extractfile(entry.source)
        if source is None:
            raise ValueError(
                f"{_UNSAFE_MEMBER_MESSAGE}: cannot read tar member {entry.path!r}"
            )
        try:
            _copy_regular_file(source, os.path.join(root, entry.path), root)
        finally:
            source.close()

    for entry in entries:
        if entry.kind != "symlink":
            continue
        assert entry.linkname is not None
        destination = os.path.join(root, entry.path)
        parent = os.path.dirname(destination)
        os.makedirs(parent, exist_ok=True)
        _ensure_within_directory(root, parent)
        _validate_symlink_target(entry.linkname, destination, root)
        if os.path.lexists(destination):
            _raise_unsafe_member(entry.path, "duplicate destination path")
        os.symlink(entry.linkname, destination)
        _ensure_within_directory(root, destination)


def _copy_zip_fs(fs: fsspec.AbstractFileSystem, temp_dir: str) -> None:
    import zipfile

    zip_file = getattr(fs, "zip", None)
    if not isinstance(zip_file, zipfile.ZipFile):
        raise ValueError("Unsupported zip filesystem object")

    entries: list[_ImportEntry] = []
    for info in zip_file.infolist():
        path = _normalize_member_path(info.filename)
        if path is None:
            continue
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"{_UNSAFE_MEMBER_MESSAGE}: zip symlinks are unsupported")
        if info.is_dir():
            entries.append(_ImportEntry(info, path, "directory"))
        else:
            entries.append(_ImportEntry(info, path, "file"))

    _validate_import_entries(entries, temp_dir)

    root = _realpath(temp_dir)
    for entry in entries:
        destination = os.path.join(root, entry.path)
        if entry.kind == "directory":
            os.makedirs(destination, exist_ok=True)
            _ensure_within_directory(root, destination)

    for entry in entries:
        if entry.kind != "file":
            continue
        with zip_file.open(entry.source, "r") as source:
            _copy_regular_file(source, os.path.join(root, entry.path), root)


def _copy_fsspec_directory(fs: fsspec.AbstractFileSystem, temp_dir: str) -> None:
    if not hasattr(fs, "walk") or not hasattr(fs, "open"):
        raise ValueError("Unsupported filesystem object")

    entries: list[_ImportEntry] = []
    for root_path, dirs, files in fs.walk("/"):
        for directory in dirs:
            source = posixpath.join(root_path, directory)
            path = _normalize_fsspec_path(source)
            if path is not None:
                entries.append(_ImportEntry(source, path, "directory"))
        for file in files:
            source = posixpath.join(root_path, file)
            path = _normalize_fsspec_path(source)
            if path is not None:
                entries.append(_ImportEntry(source, path, "file"))

    _validate_import_entries(entries, temp_dir)

    root = _realpath(temp_dir)
    for entry in entries:
        destination = os.path.join(root, entry.path)
        if entry.kind == "directory":
            os.makedirs(destination, exist_ok=True)
            _ensure_within_directory(root, destination)

    for entry in entries:
        if entry.kind != "file":
            continue
        source = t.cast(t.IO[bytes], t.cast(object, fs.open(entry.source, "rb")))
        with source:
            _copy_regular_file(source, os.path.join(root, entry.path), root)


def _safe_copy_from_fs(fs: fsspec.AbstractFileSystem, temp_dir: str) -> None:
    from fsspec.implementations.dirfs import DirFileSystem
    from fsspec.implementations.tar import TarFileSystem
    from fsspec.implementations.zip import ZipFileSystem

    if isinstance(fs, TarFileSystem):
        _copy_tar_fs(fs, temp_dir)
    elif isinstance(fs, ZipFileSystem):
        _copy_zip_fs(fs, temp_dir)
    elif isinstance(fs, DirFileSystem):
        _copy_fsspec_directory(fs, temp_dir)
    else:
        raise ValueError("Unsupported filesystem object")


class Exportable(ABC):
    _path: Path

    @staticmethod
    @abstractmethod
    def _export_ext() -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def _export_name(self) -> str:
        raise NotImplementedError

    @classmethod
    def _from_fs(cls, fs: fsspec.AbstractFileSystem) -> t.Self:
        """Create an instance from a file system."""
        temp_dir = tempfile.mkdtemp(prefix="bentoml-import-")
        try:
            _safe_copy_from_fs(fs, temp_dir)
            instance = cls.from_path(temp_dir)
            weakref.finalize(instance, safe_remove_dir, temp_dir)
            return instance
        except Exception:
            safe_remove_dir(temp_dir)
            raise

    @classmethod
    @abstractmethod
    def from_path(cls, path: PathType) -> t.Self:
        """Create an instance from a file path."""
        raise NotImplementedError

    @classmethod
    def guess_format(cls, path: str) -> str:
        _, ext = posixpath.splitext(path)

        if ext == "":
            return cls._export_ext()

        ext = ext[1:]
        if ext in [cls._export_ext(), "gz", "xz", "bz2", "tar", "zip"]:
            return ext
        else:
            return cls._export_ext()

    @classmethod
    def import_from(
        cls,
        path: str,
        input_format: str | None = None,
        *,
        protocol: str | None = None,
        user: str | None = None,
        passwd: str | None = None,
        params: dict[str, str] | None = None,
        subpath: str | None = None,
    ) -> t.Self:
        is_url = "://" in path
        if is_url and any(
            v is not None for v in [protocol, user, passwd, params, subpath]
        ):
            raise ValueError(
                "An FS URL was passed as the input path; all additional information should be passed as part of the URL."
            )
        parsedurl = urllib.parse.urlsplit(path)
        protocol = parsedurl.scheme if is_url else (protocol or "file")
        if protocol in ("osfs", "tar", "zip"):
            protocol = "file"
        if is_url:
            resource = parsedurl.netloc.rpartition("@")[2]
            subpath = parsedurl.path
            resource_url = urllib.parse.urlunsplit((protocol, *parsedurl[1:]))
        else:
            netloc = ""
            if user is not None:
                netloc += urllib.parse.quote(user)
            if passwd is not None:
                netloc += ":" + urllib.parse.quote(passwd)
            if netloc:
                netloc += "@"
            path = path.replace(os.sep, "/")
            subpath = subpath.replace(os.sep, "/") if subpath else subpath
            if subpath is None:
                if protocol == "file":
                    resource, subpath = "", path
                else:
                    resource, _, subpath = path.partition("/")
            else:
                resource = path
            netloc += resource
            url_tuple = (
                protocol,
                netloc,
                subpath,
                urllib.parse.urlencode(params or {}),
                "",
            )
            resource_url = urllib.parse.urlunsplit(url_tuple)

        if protocol not in known_implementations:
            raise ValueError(
                f"Unknown or unsupported protocol {protocol}. Some supported protocols are 'ftp', 's3', and 'file'."
            )

        if input_format is None:
            input_format = cls.guess_format(subpath)
        if protocol != "file":
            resource_url = f"filecache::{resource_url}"

        if input_format == "folder":
            from fsspec.implementations.dirfs import DirFileSystem

            fs, fspath = fsspec.url_to_fs(resource_url)
            if protocol == "file":  # Use the local file system
                return cls.from_path(fspath)
            fs = DirFileSystem(fspath, fs=fs)
        elif input_format == "zip":
            from fsspec.implementations.zip import ZipFileSystem

            fs = ZipFileSystem(resource_url)
        else:
            from fsspec.implementations.tar import TarFileSystem

            compressions = {
                cls._export_ext(): "xz",
                "gz": "gzip",
                "tar": None,
            }
            fs = TarFileSystem(
                resource_url, compression=compressions.get(input_format, input_format)
            )

        return cls._from_fs(fs)

    def export(
        self,
        path: str,
        output_format: t.Optional[str] = None,
        *,
        protocol: t.Optional[str] = None,
        user: t.Optional[str] = None,
        passwd: t.Optional[str] = None,
        params: t.Optional[t.Dict[str, str]] = None,
        subpath: t.Optional[str] = None,
    ) -> str:
        is_url = "://" in path
        if is_url and any(
            v is not None for v in [protocol, user, passwd, params, subpath]
        ):
            raise ValueError(
                "An FS URL was passed as the output path; all additional information should be passed as part of the URL."
            )
        parsedurl = urllib.parse.urlsplit(path)
        protocol = parsedurl.scheme if is_url else (protocol or "file")
        if protocol in ("osfs", "tar", "zip"):
            protocol = "file"
        if is_url:
            resource = parsedurl.netloc.rpartition("@")[2]
            subpath = parsedurl.path
            netloc = parsedurl.netloc
            query = parsedurl.query
        else:
            netloc = ""
            if user is not None:
                netloc += urllib.parse.quote(user)
            if passwd is not None:
                netloc += ":" + urllib.parse.quote(passwd)
            if netloc:
                netloc += "@"
            path = path.replace(os.sep, "/")
            subpath = subpath.replace(os.sep, "/") if subpath else subpath
            if subpath is None:
                if protocol == "file":
                    resource, subpath = "", path
                else:
                    resource, _, subpath = path.partition("/")
            else:
                resource = path
            netloc += resource
            query = urllib.parse.urlencode(params or {})

        if protocol not in known_implementations:
            raise ValueError(
                f"Unknown or unsupported protocol {protocol}. Some supported protocols are 'ftp', 's3', and 'file'."
            )

        if output_format is None:
            output_format = self.guess_format(subpath)

        is_dir = (
            not subpath
            or subpath.endswith("/")
            or protocol == "file"
            and os.path.isdir(subpath)
        )

        if output_format != "folder" and is_dir:
            subpath = posixpath.join(
                subpath or "", f"{self._export_name}.{output_format}"
            )
        if output_format == "folder" and not subpath.endswith("/"):
            subpath += "/"

        if output_format == self._export_ext() and not subpath.endswith(
            f".{self._export_ext()}"
        ):
            logging.info(
                f"Adding {self._export_ext()} because ext is {posixpath.splitext(subpath)[1]}"
            )
            subpath += f".{self._export_ext()}"
        resource_url = urllib.parse.urlunsplit((protocol, netloc, subpath, query, ""))

        fs, fspath = fsspec.url_to_fs(resource_url)
        if output_format == "folder":
            fs.put(str(self._path), fspath, recursive=True)
        else:
            temp_name = tempfile.mktemp(prefix="bentoml-export-")
            compressed = self._compress(temp_name, output_format)
            try:
                fs.put(compressed, fspath)
            finally:
                os.remove(compressed)
        return fspath

    def _compress(self, path: str, output_format: str) -> str:
        formats = {
            "gz": "gztar",
            "xz": "xztar",
            "bz2": "bztar",
            self._export_ext(): "xztar",
        }
        if output_format in ["gz", "xz", "bz2", "tar", self._export_ext(), "zip"]:
            return shutil.make_archive(
                path, formats.get(output_format, output_format), str(self._path)
            )
        else:
            raise ValueError(f"Unsupported format {output_format}")
