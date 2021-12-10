import abc, typing
from typing import Optional, Text
import six
from .osfs import OSFS

if typing.TYPE_CHECKING: ...
__all__ = [
    "UserDataFS",
    "UserConfigFS",
    "SiteDataFS",
    "SiteConfigFS",
    "UserCacheFS",
    "UserLogFS",
]

class _CopyInitMeta(abc.ABCMeta):
    def __new__(mcls, classname, bases, cls_dict): ...

@six.add_metaclass(_CopyInitMeta)
class _AppFS(OSFS):
    app_dir: Text = None
    def __init__(
        self,
        appname: Text,
        author: Optional[Text] = ...,
        version: Optional[Text] = ...,
        roaming: bool = ...,
        create: bool = ...,
    ) -> None: ...
    def __repr__(self) -> Text: ...
    def __str__(self) -> Text: ...

class UserDataFS(_AppFS):
    app_dir = ...

class UserConfigFS(_AppFS):
    app_dir = ...

class UserCacheFS(_AppFS):
    app_dir = ...

class SiteDataFS(_AppFS):
    app_dir = ...

class SiteConfigFS(_AppFS):
    app_dir = ...

class UserLogFS(_AppFS):
    app_dir = ...
