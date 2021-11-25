

import abc
import typing
from typing import Optional, Text

import six

from .osfs import OSFS

"""Manage filesystems in platform-specific application directories.

These classes abstract away the different requirements for user data
across platforms, which vary in their conventions. They are all
subclasses of `~fs.osfs.OSFS`.

"""
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
    """A metaclass that performs a hard copy of the `__init__`.

    This is a fix for Sphinx, which is a pain to configure in a way that
    it documents the ``__init__`` method of a class when it is inherited.
    Copying ``__init__`` makes it think it is not inherited, and let us
    share the documentation between all the `_AppFS` subclasses.

    """

    def __new__(mcls, classname, bases, cls_dict): ...

@six.add_metaclass(_CopyInitMeta)
class _AppFS(OSFS):
    """Abstract base class for an app FS."""

    app_dir: Text = None
    def __init__(
        self,
        appname: Text,
        author: Optional[Text] = ...,
        version: Optional[Text] = ...,
        roaming: bool = ...,
        create: bool = ...,
    ) -> None:
        """Create a new application-specific filesystem.

        Arguments:
            appname (str): The name of the application.
            author (str): The name of the author (used on Windows).
            version (str): Optional version string, if a unique location
                per version of the application is required.
            roaming (bool): If `True`, use a *roaming* profile on
                Windows.
            create (bool): If `True` (the default) the directory
                will be created if it does not exist.

        """
        ...
    def __repr__(self) -> Text: ...
    def __str__(self) -> Text: ...

class UserDataFS(_AppFS):
    """A filesystem for per-user application data.

    May also be opened with
    ``open_fs('userdata://appname:author:version')``.

    """

    app_dir = ...

class UserConfigFS(_AppFS):
    """A filesystem for per-user config data.

    May also be opened with
    ``open_fs('userconf://appname:author:version')``.

    """

    app_dir = ...

class UserCacheFS(_AppFS):
    """A filesystem for per-user application cache data.

    May also be opened with
    ``open_fs('usercache://appname:author:version')``.

    """

    app_dir = ...

class SiteDataFS(_AppFS):
    """A filesystem for application site data.

    May also be opened with
    ``open_fs('sitedata://appname:author:version')``.

    """

    app_dir = ...

class SiteConfigFS(_AppFS):
    """A filesystem for application config data.

    May also be opened with
    ``open_fs('siteconf://appname:author:version')``.

    """

    app_dir = ...

class UserLogFS(_AppFS):
    """A filesystem for per-user application log data.

    May also be opened with
    ``open_fs('userlog://appname:author:version')``.

    """

    app_dir = ...
