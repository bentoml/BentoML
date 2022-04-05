import re
import uuid
import base64
import typing as t
import logging

import fs
import attr

from .utils import bentoml_cattr
from ..exceptions import InvalidArgument
from ..exceptions import BentoMLException

logger = logging.getLogger(__name__)

tag_fmt = "[a-z0-9]([-._a-z0-9]*[a-z0-9])?"
tag_max_length = 63
tag_max_length_error_msg = (
    "a tag's name or version must be at most {tag_max_length} characters in length"
)
tag_invalid_error_msg = "a tag's name or version must consist of alphanumeric characters, '_', '-', or '.', and must start and end with an alphanumeric character"
tag_regex = re.compile(f"^{tag_fmt}$")


def validate_tag_str(value: str):
    """
    Validate that a tag value (either name or version) is a simple string that:
        * Must be at most 63 characters long,
        * Begin and end with an alphanumeric character (`[a-z0-9A-Z]`), and
        * May contain dashes (`-`), underscores (`_`) dots (`.`), or alphanumerics
          between.
    """
    errors = []
    if len(value) > tag_max_length:
        errors.append(tag_max_length_error_msg)
    if tag_regex.match(value) is None:
        errors.append(tag_invalid_error_msg)

    if errors:
        raise InvalidArgument(f"{value} is not a valid tag: " + ", and ".join(errors))


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
        # cut out the time_hi and node bits of the uuid
        ver_bytes = ver_bytes[:6] + ver_bytes[8:12]
        encoded_ver = base64.b32encode(ver_bytes)

        return Tag(self.name, encoded_ver.decode("ascii").lower())

    def path(self) -> str:
        if self.version is None:
            return self.name
        return fs.path.combine(self.name, self.version)

    def latest_path(self) -> str:
        return fs.path.combine(self.name, "latest")


bentoml_cattr.register_structure_hook(Tag, lambda d, _: Tag.from_taglike(d))  # type: ignore[misc]
bentoml_cattr.register_unstructure_hook(Tag, lambda tag: str(tag))
