import io
import os
import uuid
import base64
import typing as t
import urllib
import logging
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass

import fs
import attr
import cattr

from ..exceptions import BentoMLException
from .utils.validation import validate_tag_str
from .utils.dataclasses import json_serializer

logger = logging.getLogger(__name__)

BATCH_HEADER = "Bentoml-Is-Batch-Request"

# For non latin1 characters: https://tools.ietf.org/html/rfc8187
# Also https://github.com/benoitc/gunicorn/issues/1778
HEADER_CHARSET = "latin1"

JSON_CHARSET = "utf-8"

if TYPE_CHECKING:
    PathType = t.Union[str, os.PathLike[str]]
else:
    PathType = t.Union[str, os.PathLike]

JSONSerializable = t.NewType("JSONSerializable", object)


@attr.define
class Tag:
    name: str
    version: t.Optional[str]

    def __init__(self, name: str, version: t.Optional[str] = None):
        lname = name.lower()
        if name != lname:
            logger.warning(f"converting {name} to lowercase: {lname}")

        validate_tag_str(lname)

        self.name = lname

        if version is not None:
            lversion = version.lower()
            if version != lversion:
                logger.warning(f"converting {version} to lowercase: {lversion}")
            validate_tag_str(lversion)
            self.version = lversion
        else:
            self.version = None

    def __str__(self):
        if self.version is None:
            return self.name
        else:
            return f"{self.name}:{self.version}"

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)}, version={repr(self.version)})"

    def __eq__(self, other: "Tag") -> bool:
        return self.name == other.name and self.version == other.version

    def __lt__(self, other: "Tag") -> bool:
        if self.name == other.name:
            if other.version is None:
                return False
            if self.version is None:
                return True
            return self.version < other.version
        return self.name < other.name

    def __hash__(self) -> int:
        return hash((self.name, self.version))

    @classmethod
    def from_taglike(cls, taglike: t.Union["Tag", str]) -> "Tag":
        if isinstance(taglike, Tag):
            return taglike
        return cls.from_str(taglike)

    @classmethod
    def from_str(cls, tag_str: str) -> "Tag":
        if ":" not in tag_str:
            return cls(tag_str, None)
        try:
            name, _, version = tag_str.partition(":")
            if not version:
                # in case users mistakenly define "bento:"
                raise BentoMLException(
                    f"{tag_str} contains trailing ':'. Maybe you meant to use `{tag_str}:latest`?"
                )
            return cls(name, version)
        except ValueError:
            raise BentoMLException(f"Invalid {cls.__name__} {tag_str}")

    def make_new_version(self) -> "Tag":
        if self.version is not None:
            raise ValueError(
                "tried to run 'make_new_version' on a Tag that already has a version"
            )
        ver_bytes = bytearray(uuid.uuid1().bytes)
        # replace 3 bits of the version with the end of the uuid
        ver_bytes[6] = (ver_bytes[6] & 0x1F) | ((ver_bytes[-1] & 0x7) << 5)
        # last 7 bytes of the base32 encode are padding
        encoded_ver = base64.b32encode(ver_bytes)[:-7]

        return Tag(self.name, encoded_ver.decode("ascii").lower())

    def path(self) -> str:
        if self.version is None:
            return self.name
        return fs.path.combine(self.name, self.version)

    def latest_path(self) -> str:
        return fs.path.combine(self.name, "latest")


cattr.register_structure_hook(Tag, lambda d, t: Tag.from_taglike(d))

cattr.register_structure_hook(
    datetime, lambda d, t: d if isinstance(d, datetime) else datetime.fromisoformat(d)
)


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
    def stream(self) -> t.BinaryIO:
        if self._stream is not None:
            pass
        elif self.bytes_ is not None:
            self._stream = io.BytesIO(self.bytes_)
        elif self.uri is not None:
            self._stream = open(self.path, "rb")
        else:
            return io.BytesIO()
        return self._stream

    def read(self, size: int = -1):
        # TODO: also write to log
        return self.stream.read(size)

    def seek(self, pos: int):
        return self.stream.seek(pos)

    def tell(self):
        return self.stream.tell()

    def close(self):
        if self._stream is not None:
            self._stream.close()
