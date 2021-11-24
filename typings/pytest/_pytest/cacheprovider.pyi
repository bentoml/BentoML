
from pathlib import Path
from typing import Generator, List, Optional, Set, Union

import attr
import py
from _pytest import nodes
from _pytest.compat import final
from _pytest.config import Config, ExitCode, hookimpl
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest, fixture
from _pytest.main import Session
from _pytest.reports import TestReport

from .reports import CollectReport

"""Implementation of the cache provider."""
README_CONTENT = ...
CACHEDIR_TAG_CONTENT = ...
@final
@attr.s(init=False)
class Cache:
    _cachedir = ...
    _config = ...
    _CACHE_PREFIX_DIRS = ...
    _CACHE_PREFIX_VALUES = ...
    def __init__(self, cachedir: Path, config: Config, *, _ispytest: bool = ...) -> None:
        ...
    
    @classmethod
    def for_config(cls, config: Config, *, _ispytest: bool = ...) -> Cache:
        """Create the Cache instance for a Config.

        :meta private:
        """
        ...
    
    @classmethod
    def clear_cache(cls, cachedir: Path, _ispytest: bool = ...) -> None:
        """Clear the sub-directories used to hold cached directories and values.

        :meta private:
        """
        ...
    
    @staticmethod
    def cache_dir_from_config(config: Config, *, _ispytest: bool = ...) -> Path:
        """Get the path to the cache directory for a Config.

        :meta private:
        """
        ...
    
    def warn(self, fmt: str, *, _ispytest: bool = ..., **args: object) -> None:
        """Issue a cache warning.

        :meta private:
        """
        ...
    
    def makedir(self, name: str) -> py.path.local:
        """Return a directory path object with the given name.

        If the directory does not yet exist, it will be created. You can use
        it to manage files to e.g. store/retrieve database dumps across test
        sessions.

        :param name:
            Must be a string not containing a ``/`` separator.
            Make sure the name contains your plugin or application
            identifiers to prevent clashes with other cache users.
        """
        ...
    
    def get(self, key: str, default): # -> Any:
        """Return the cached value for the given key.

        If no value was yet cached or the value cannot be read, the specified
        default is returned.

        :param key:
            Must be a ``/`` separated value. Usually the first
            name is the name of your plugin or your application.
        :param default:
            The value to return in case of a cache-miss or invalid cache value.
        """
        ...
    
    def set(self, key: str, value: object) -> None:
        """Save value for the given key.

        :param key:
            Must be a ``/`` separated value. Usually the first
            name is the name of your plugin or your application.
        :param value:
            Must be of any combination of basic python types,
            including nested types like lists of dictionaries.
        """
        ...
    


class LFPluginCollWrapper:
    def __init__(self, lfplugin: LFPlugin) -> None:
        ...
    
    @hookimpl(hookwrapper=True)
    def pytest_make_collect_report(self, collector: nodes.Collector): # -> Generator[None, None, None]:
        ...
    


class LFPluginCollSkipfiles:
    def __init__(self, lfplugin: LFPlugin) -> None:
        ...
    
    @hookimpl
    def pytest_make_collect_report(self, collector: nodes.Collector) -> Optional[CollectReport]:
        ...
    


class LFPlugin:
    """Plugin which implements the --lf (run last-failing) option."""
    def __init__(self, config: Config) -> None:
        ...
    
    def get_last_failed_paths(self) -> Set[Path]:
        """Return a set with all Paths()s of the previously failed nodeids."""
        ...
    
    def pytest_report_collectionfinish(self) -> Optional[str]:
        ...
    
    def pytest_runtest_logreport(self, report: TestReport) -> None:
        ...
    
    def pytest_collectreport(self, report: CollectReport) -> None:
        ...
    
    @hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, config: Config, items: List[nodes.Item]) -> Generator[None, None, None]:
        ...
    
    def pytest_sessionfinish(self, session: Session) -> None:
        ...
    


class NFPlugin:
    """Plugin which implements the --nf (run new-first) option."""
    def __init__(self, config: Config) -> None:
        ...
    
    @hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, items: List[nodes.Item]) -> Generator[None, None, None]:
        ...
    
    def pytest_sessionfinish(self) -> None:
        ...
    


def pytest_addoption(parser: Parser) -> None:
    ...

def pytest_cmdline_main(config: Config) -> Optional[Union[int, ExitCode]]:
    ...

@hookimpl(tryfirst=True)
def pytest_configure(config: Config) -> None:
    ...

@fixture
def cache(request: FixtureRequest) -> Cache:
    """Return a cache object that can persist state between testing sessions.

    cache.get(key, default)
    cache.set(key, value)

    Keys must be ``/`` separated strings, where the first part is usually the
    name of your plugin or application to avoid clashes with other cache users.

    Values can be any object handled by the json stdlib module.
    """
    ...

def pytest_report_header(config: Config) -> Optional[str]:
    """Display cachedir with --cache-show and if non-default."""
    ...

def cacheshow(config: Config, session: Session) -> int:
    ...

