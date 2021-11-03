import io
import logging
import os
import pathlib
import typing as t
import urllib
import urllib.parse
import urllib.request
from dataclasses import dataclass

import fs

from bentoml.exceptions import BentoMLException

from .utils.dataclasses import json_serializer
from .utils.validation import validate_tag_name_str, validate_version_str

logger = logging.getLogger(__name__)

BATCH_HEADER = "Bentoml-Is-Batch-Request"

# For non latin1 characters: https://tools.ietf.org/html/rfc8187
# Also https://github.com/benoitc/gunicorn/issues/1778
HEADER_CHARSET = "latin1"

JSON_CHARSET = "utf-8"

PathType = t.Union[str, os.PathLike, pathlib.Path]

JSONSerializable = t.NewType("JSONSerializable", object)


class Tag:
    name: str
    version: t.Optional[str]

    def __init__(self, name: str, version: t.Optional[str]):
        lname = name.lower()
        if name != lname:
            logger.warning(f"converting {name} to lowercase: {lname}")

        validate_tag_name_str(lname)

        if version is not None:
            validate_version_str(version)

        self.name = lname
        self.version = version

    def __str__(self):
        if self.version is None:
            return self.name
        else:
            return f"{self.name}:{self.version}"

    def __lt__(self, other):
        if self.name == other.name:
            return self.version < other.version
        return self.name < other.name

    @classmethod
    def from_taglike(cls, taglike: t.Union["Tag", str]) -> "Tag":
        if isinstance(taglike, Tag):
            return taglike
        if isinstance(taglike, str):
            return cls.from_str(taglike)

        raise TypeError(f"can't make a Tag from {type(taglike)}")

    @classmethod
    def from_str(cls, tag_str: str) -> "Tag":
        if ":" not in tag_str:
            return cls(tag_str, None)
        try:
            name, version = tag_str.split(":")
            if not version:
                # in case users mistakenly define "bento:"
                raise BentoMLException(
                    f"{tag_str} contains trailing ':'. Maybe you meant to use `{tag_str}:latest`?"
                )
            return cls(name, version)
        except ValueError:
            raise BentoMLException(f"Invalid {cls.__name__} {tag_str}")

    def path(self) -> str:
        if self.version is None:
            return self.name
        return fs.path.combine(self.name, self.version)

    def latest_path(self) -> str:
        return fs.path.combine(self.name, "latest")


@json_serializer(fields=["uri", "name"], compat=True)
@dataclass(frozen=False)
class FileLike:
    """
    An universal lazy-loading wrapper for file-like objects.
    It accepts URI, file path or bytes and provides interface like opened file object.

    Class attributes:

    - bytes (`bytes`, `optional`):
    - uri (`str`, `optional`):
        The set of possible uris is:

        - :code:`file:///home/user/input.json`
        - :code:`http://site.com/input.csv` (Not implemented)
        - :code:`https://site.com/input.csv` (Not implemented)

    - name (`str`, `optional`, default to :obj:`None`)

    """

    bytes_: t.Optional[bytes] = None
    uri: t.Optional[str] = None
    name: t.Optional[str] = None

    _stream: t.Optional[t.BinaryIO] = None

    def __post_init__(self):
        if self.name is None:
            if self._stream is not None:
                self.name = getattr(self._stream, "name", None)
            elif self.uri is not None:
                p = urllib.parse.urlparse(self.uri)  # type: ignore
                if p.scheme and p.scheme != "file":
                    raise NotImplementedError(
                        f"{self.__class__} now supports scheme 'file://' only"
                    )
                _, self.name = os.path.split(self.path)

    @property
    def path(self):
        r"""
        supports:

            /home/user/file
            C:\Python27\Scripts\pip.exe
            \\localhost\c$\WINDOWS\clock.avi
            \\networkstorage\homes\user

        .. note::
            https://stackoverflow.com/a/61922504/3089381
        """
        parsed = urllib.parse.urlparse(self.uri)  # type: ignore
        raw_path = urllib.request.url2pathname(urllib.parse.unquote(parsed.path))  # type: ignore # noqa: LN001
        host = "{0}{0}{mnt}{0}".format(os.path.sep, mnt=parsed.netloc)
        path = os.path.abspath(os.path.join(host, raw_path))
        return path

    @property
    def stream(self):
        if self._stream is not None:
            pass
        elif self.bytes_ is not None:
            self._stream = io.BytesIO(self.bytes_)
        elif self.uri is not None:
            self._stream = open(self.path, "rb")
        else:
            return io.BytesIO()
        return self._stream

    def read(self, size=-1):
        # TODO: also write to log
        return self.stream.read(size)

    def seek(self, pos):
        return self.stream.seek(pos)

    def tell(self):
        return self.stream.tell()

    def close(self):
        if self._stream is not None:
            self._stream.close()

    def __del__(self):
        if (
            hasattr(self, "_stream")
            and self._stream is not None
            and not self._stream.closed
        ):
            self._stream.close()
