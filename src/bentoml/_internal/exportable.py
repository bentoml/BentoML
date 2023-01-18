from __future__ import annotations

import os
import re
import typing as t
import logging
import urllib.parse
from abc import ABC
from abc import abstractmethod

import fs
import fs.copy
import fs.errors
import fs.mirror
import fs.opener
import fs.tempfs
import fs.opener.errors
from fs import open_fs
from fs.base import FS

T = t.TypeVar("T", bound="Exportable")


uriSchemeRe = re.compile(r".*[^\\](?=://)")


class Exportable(ABC):
    @staticmethod
    @abstractmethod
    def _export_ext() -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def _export_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def _fs(self) -> FS:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_fs(cls: t.Type[T], item_fs: FS) -> T:
        raise NotImplementedError

    @classmethod
    def guess_format(cls, path: str) -> str:
        _, ext = fs.path.splitext(path)

        if ext == "":
            return cls._export_ext()

        ext = ext[1:]
        if ext in [cls._export_ext(), "gz", "xz", "bz2", "tar", "zip"]:
            return ext
        else:
            return cls._export_ext()

    @classmethod
    def import_from(
        cls: t.Type[T],
        path: str,
        input_format: t.Optional[str] = None,
        *,
        protocol: t.Optional[str] = None,
        user: t.Optional[str] = None,
        passwd: t.Optional[str] = None,
        params: t.Optional[t.Dict[str, str]] = None,
        subpath: t.Optional[str] = None,
    ) -> T:
        try:
            parsedurl = fs.opener.parse(path)
        except fs.opener.errors.ParseError:
            if protocol is None:
                protocol = "osfs"
                resource: str = path if os.sep == "/" else path.replace(os.sep, "/")
            else:
                resource: str = ""
        else:
            if any(v is not None for v in [protocol, user, passwd, params, subpath]):
                raise ValueError(
                    "An FS URL was passed as the output path; all additional information should be passed as part of the URL."
                )

            protocol = parsedurl.protocol  # type: ignore (FS types)
            user = parsedurl.username  # type: ignore (FS types)
            passwd = parsedurl.password  # type: ignore (FS types)
            resource: str = parsedurl.resource  # type: ignore (FS types)
            params = parsedurl.params  # type: ignore (FS types)
            subpath = parsedurl.path  # type: ignore (FS types)

        if protocol not in fs.opener.registry.protocols:  # type: ignore (FS types)
            if protocol == "s3":
                raise ValueError(
                    "Tried to open an S3 url but the protocol is not registered; did you 'pip install \"bentoml[aws]\"'?"
                )
            else:
                raise ValueError(
                    f"Unknown or unsupported protocol {protocol}. Some supported protocols are 'ftp', 's3', and 'osfs'."
                )

        isOSPath = protocol in [
            "temp",
            "osfs",
            "userdata",
            "userconf",
            "sitedata",
            "siteconf",
            "usercache",
            "userlog",
        ]

        isCompressedPath = protocol in ["tar", "zip"]

        if input_format is not None:
            if isCompressedPath:
                raise ValueError(
                    f"A {protocol} path was passed along with an input format ({input_format}); only pass one or the other."
                )
        else:
            # try to guess output format if it hasn't been passed
            input_format = cls.guess_format(resource if subpath is None else subpath)

        if subpath is None:
            if isCompressedPath:
                subpath = ""
            else:
                resource, subpath = fs.path.split(resource)

        if isOSPath:
            # we can safely ignore user / passwd / params
            # get the path on the system so we can skip the copy step
            try:
                rsc_dir = fs.open_fs(f"{protocol}://{resource}")
                # if subpath is root, we can just open directly
                if (subpath == "" or subpath == "/") and input_format == "folder":
                    return cls.from_fs(rsc_dir)

                # if subpath is specified, we need to create our own temp fs and mirror that subpath
                path = rsc_dir.getsyspath(subpath)
                res = cls._from_compressed(path, input_format)
            except fs.errors.CreateFailed:
                raise ValueError("Directory does not exist")
        else:
            userblock = ""
            if user is not None or passwd is not None:
                if user is not None:
                    userblock += urllib.parse.quote(user)
                if passwd is not None:
                    userblock += ":" + urllib.parse.quote(passwd)
                userblock += "@"

            resource = urllib.parse.quote(resource)

            query = ""
            if params is not None:
                query = "?" + urllib.parse.urlencode(params)

            input_uri = f"{protocol}://{userblock}{resource}{query}"

            if isCompressedPath:
                input_fs = fs.open_fs(input_uri)
                res = cls.from_fs(input_fs)
            else:
                filename = fs.path.basename(subpath)
                tempfs = fs.tempfs.TempFS(identifier=f"bentoml-{filename}-import")
                if input_format == "folder":
                    input_fs = fs.open_fs(input_uri)
                    fs.copy.copy_dir(input_fs, subpath, tempfs, filename)
                else:
                    input_fs = fs.open_fs(input_uri)
                    fs.copy.copy_file(input_fs, subpath, tempfs, filename)

                res = cls._from_compressed(tempfs.getsyspath(filename), input_format)

            input_fs.close()

        return res

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
        try:
            parsedurl = fs.opener.parse(path)
        except fs.opener.errors.ParseError:  # type: ignore (incomplete FS types)
            if protocol is None:
                protocol = "osfs"
                resource = path if os.sep == "/" else path.replace(os.sep, "/")
            else:
                resource = ""
        else:
            if any(v is not None for v in [protocol, user, passwd, params, subpath]):
                raise ValueError(
                    "An FS URL was passed as the output path; all additional information should be passed as part of the URL."
                )

            protocol = parsedurl.protocol  # type: ignore (incomplete FS types)
            user = parsedurl.username  # type: ignore (incomplete FS types)
            passwd = parsedurl.password  # type: ignore (incomplete FS types)
            resource: str = parsedurl.resource  # type: ignore (incomplete FS types)
            params = parsedurl.params  # type: ignore (incomplete FS types)
            subpath = parsedurl.path  # type: ignore (incomplete FS types)

        if protocol not in fs.opener.registry.protocols:  # type: ignore (incomplete FS types)
            if protocol == "s3":
                raise ValueError(
                    "Tried to open an S3 url but the protocol is not registered; did you install fs-s3fs?"
                )
            else:
                raise ValueError(
                    f"Unknown or unsupported protocol {protocol}. Some supported protocols are 'ftp', 's3', and 'osfs'."
                )

        isOSPath = protocol in [
            "temp",
            "osfs",
            "userdata",
            "userconf",
            "sitedata",
            "siteconf",
            "usercache",
            "userlog",
        ]

        isCompressedPath = protocol in ["tar", "zip"]

        if output_format is not None:
            if isCompressedPath:
                raise ValueError(
                    f"A {protocol} path was passed along with an output format ({output_format}); only pass one or the other."
                )
        else:
            # try to guess output format if it hasn't been passed
            output_format = self.guess_format(resource if subpath is None else subpath)

        # use combine here instead of join because we don't want to fold subpath into path.
        fullpath = resource if subpath is None else fs.path.combine(resource, subpath)

        if fullpath == "" or (not output_format == "folder" and fullpath[-1] == "/"):
            subpath = "" if subpath is None else subpath
            # if the path has no filename or is empty, we generate a filename from the bento tag
            subpath = fs.path.join(subpath, self._export_name)
            if output_format in ["gz", "xz", "bz2", "tar", "zip"]:
                subpath += "." + output_format
        elif isOSPath:
            try:
                dirfs = fs.open_fs(path)
            except fs.errors.CreateFailed:
                pass
            else:
                if subpath is None or dirfs.isdir(subpath):
                    subpath = "" if subpath is None else subpath
                    # path exists and is a directory, we will create a file inside that directory
                    subpath = fs.path.join(subpath, self._export_name)
                    if output_format in ["gz", "xz", "bz2", "tar", "zip"]:
                        subpath += "." + output_format

        if subpath is None:
            if isCompressedPath:
                subpath = ""
            else:
                resource, subpath = fs.path.split(resource)

        # as a special case, if the output format is the default we will always append the relevant
        # extension if the output filename does not have it
        if (
            output_format == self._export_ext()
            and fs.path.splitext(subpath)[1] != "." + self._export_ext()
        ):
            logging.info(
                f"adding {self._export_ext} because ext is {fs.path.splitext(subpath)[1]}"
            )
            subpath += "." + self._export_ext()

        if isOSPath:
            # we can safely ignore user / passwd / params
            # get the path on the system so we can skip the copy step
            try:
                rsc_dir = fs.open_fs(f"{protocol}://{resource}")
                path = rsc_dir.getsyspath(subpath)
                self._compress(path, output_format)
            except fs.errors.CreateFailed:
                raise ValueError(
                    f"Output directory '{protocol}://{resource}' does not exist."
                )
        else:
            userblock = ""
            if user is not None or passwd is not None:
                if user is not None:
                    userblock += urllib.parse.quote(user)
                if passwd is not None:
                    userblock += ":" + urllib.parse.quote(passwd)
                userblock += "@"

            resource = urllib.parse.quote(resource)

            query = ""
            if params is not None:
                query = "?" + urllib.parse.urlencode(params)

            dest_uri = f"{protocol}://{userblock}{resource}{query}"

            if isCompressedPath:
                destfs = fs.open_fs(dest_uri, writeable=True, create=True)
                fs.mirror.mirror(self._fs, destfs, copy_if_newer=False)
            else:
                tempfs = fs.tempfs.TempFS(
                    identifier=f"bentoml-{self._export_name}-export"
                )
                filename = fs.path.basename(subpath)
                self._compress(tempfs.getsyspath(filename), output_format)

                if output_format == "folder":
                    destfs = fs.open_fs(dest_uri)
                    fs.copy.copy_dir(tempfs, filename, destfs, subpath)
                else:
                    destfs = fs.open_fs(dest_uri)
                    fs.copy.copy_file(tempfs, filename, destfs, subpath)

                tempfs.close()

            destfs.close()

        return path

    def _compress(self, path: str, output_format: str):
        if output_format in ["gz", "xz", "bz2", "tar", self._export_ext()]:
            from fs.tarfs import WriteTarFS

            compression = output_format
            if compression == "tar":
                compression = None
            if compression == self._export_ext():
                compression = "xz"
            out_fs = WriteTarFS(path, compression)
        elif output_format == "zip":
            from fs.zipfs import WriteZipFS

            out_fs = WriteZipFS(path)
        elif output_format == "folder":
            out_fs: FS = open_fs(path, writeable=True, create=True)
        else:
            raise ValueError(f"Unsupported format {output_format}")

        fs.mirror.mirror(self._fs, out_fs, copy_if_newer=False)
        out_fs.close()

    @classmethod
    def _from_compressed(cls: t.Type[T], path: str, input_format: str) -> T:
        if input_format in ["gz", "xz", "bz2", "tar", cls._export_ext()]:
            from fs.tarfs import ReadTarFS

            compression = input_format
            if compression == "tar":
                compression = None
            if compression == cls._export_ext():
                compression = "xz"
            ret_fs = ReadTarFS(path)
        elif input_format == "zip":
            from fs.zipfs import ReadZipFS

            ret_fs = ReadZipFS(path)
        elif input_format == "folder":
            ret_fs: FS = open_fs(path)
        else:
            raise ValueError(f"Unsupported format {input_format}")

        return cls.from_fs(ret_fs)
