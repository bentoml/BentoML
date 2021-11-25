

from __future__ import annotations

import collections
import itertools
import re
import warnings
from typing import Callable, Iterator, SupportsInt, Tuple, Union

__all__ = ["parse", "Version", "LegacyVersion", "InvalidVersion", "VERSION_PATTERN"]
class InfinityType:
    def __repr__(self) -> str:
        ...
    
    def __hash__(self) -> int:
        ...
    
    def __lt__(self, other: object) -> bool:
        ...
    
    def __le__(self, other: object) -> bool:
        ...
    
    def __eq__(self, other: object) -> bool:
        ...
    
    def __ne__(self, other: object) -> bool:
        ...
    
    def __gt__(self, other: object) -> bool:
        ...
    
    def __ge__(self, other: object) -> bool:
        ...
    
    def __neg__(self: object) -> NegativeInfinityType:
        ...
    


Infinity = ...
class NegativeInfinityType:
    def __repr__(self) -> str:
        ...
    
    def __hash__(self) -> int:
        ...
    
    def __lt__(self, other: object) -> bool:
        ...
    
    def __le__(self, other: object) -> bool:
        ...
    
    def __eq__(self, other: object) -> bool:
        ...
    
    def __ne__(self, other: object) -> bool:
        ...
    
    def __gt__(self, other: object) -> bool:
        ...
    
    def __ge__(self, other: object) -> bool:
        ...
    
    def __neg__(self: object) -> InfinityType:
        ...
    


NegativeInfinity = ...
InfiniteTypes = Union[InfinityType, NegativeInfinityType]
PrePostDevType = Union[InfiniteTypes, Tuple[str, int]]
SubLocalType = Union[InfiniteTypes, int, str]
LocalType = Union[NegativeInfinityType, Tuple[Union[SubLocalType, Tuple[SubLocalType, str], Tuple[NegativeInfinityType, SubLocalType]],, ...],],
CmpKey = Tuple[int, Tuple[int, ...], PrePostDevType, PrePostDevType, PrePostDevType, LocalType]
LegacyCmpKey = Tuple[int, Tuple[str, ...]]
VersionComparisonMethod = Callable[[Union[CmpKey, LegacyCmpKey], Union[CmpKey, LegacyCmpKey]], bool]
_Version = ...
def parse(version: str) -> LegacyVersion | Version:
    """
    Parse the given version string and return either a :class:`Version` object
    or a :class:`LegacyVersion` object depending on if the given version is
    a valid PEP 440 version or a legacy version.
    """
    ...

class InvalidVersion(ValueError):
    """
    An invalid version was found, users should refer to PEP 440.
    """
    ...


class _BaseVersion:
    _key: CmpKey | LegacyCmpKey
    def __hash__(self) -> int:
        ...
    
    def __lt__(self, other: _BaseVersion) -> bool:
        ...
    
    def __le__(self, other: _BaseVersion) -> bool:
        ...
    
    def __eq__(self, other: object) -> bool:
        ...
    
    def __ge__(self, other: _BaseVersion) -> bool:
        ...
    
    def __gt__(self, other: _BaseVersion) -> bool:
        ...
    
    def __ne__(self, other: object) -> bool:
        ...
    


class LegacyVersion(_BaseVersion):
    def __init__(self, version: str) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @property
    def public(self) -> str:
        ...
    
    @property
    def base_version(self) -> str:
        ...
    
    @property
    def epoch(self) -> int:
        ...
    
    @property
    def release(self) -> None:
        ...
    
    @property
    def pre(self) -> None:
        ...
    
    @property
    def post(self) -> None:
        ...
    
    @property
    def dev(self) -> None:
        ...
    
    @property
    def local(self) -> None:
        ...
    
    @property
    def is_prerelease(self) -> bool:
        ...
    
    @property
    def is_postrelease(self) -> bool:
        ...
    
    @property
    def is_devrelease(self) -> bool:
        ...
    


_legacy_version_component_re = ...
_legacy_version_replacement_map = ...
VERSION_PATTERN = ...
class Version(_BaseVersion):
    _regex = ...
    def __init__(self, version: str) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def __str__(self) -> str:
        ...
    
    @property
    def epoch(self) -> int:
        ...
    
    @property
    def release(self) -> tuple[int, ...]:
        ...
    
    @property
    def pre(self) -> tuple[str, int] | None:
        ...
    
    @property
    def post(self) -> int | None:
        ...
    
    @property
    def dev(self) -> int | None:
        ...
    
    @property
    def local(self) -> str | None:
        ...
    
    @property
    def public(self) -> str:
        ...
    
    @property
    def base_version(self) -> str:
        ...
    
    @property
    def is_prerelease(self) -> bool:
        ...
    
    @property
    def is_postrelease(self) -> bool:
        ...
    
    @property
    def is_devrelease(self) -> bool:
        ...
    
    @property
    def major(self) -> int:
        ...
    
    @property
    def minor(self) -> int:
        ...
    
    @property
    def micro(self) -> int:
        ...
    


_local_version_separators = ...
