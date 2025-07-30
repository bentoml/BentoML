from __future__ import annotations

import logging
import os
import posixpath
import shutil
import tempfile
import typing as t
import urllib.parse
import weakref
from abc import ABC
from abc import abstractmethod
from pathlib import Path

import fsspec
from fsspec.registry import known_implementations

from .types import PathType
from .utils.filesystem import safe_remove_dir

T = t.TypeVar("T", bound="Exportable")


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
        fs.get("/", temp_dir, recursive=True)
        instance = cls.from_path(temp_dir)
        weakref.finalize(instance, safe_remove_dir, temp_dir)
        return instance

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
