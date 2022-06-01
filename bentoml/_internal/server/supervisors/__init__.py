from typing import Type
from typing import TYPE_CHECKING

from .baseplugin import ReloaderPlugin

if TYPE_CHECKING:
    ServiceReloaderPlugin: Type[ReloaderPlugin]
else:
    try:
        from .watchfilesplugin import WatchFilesPlugin as ServiceReloaderPlugin
    except ImportError:
        from .statsplugin import StatsPlugin as ServiceReloaderPlugin

__all__ = ["ServiceReloaderPlugin"]
