from yatai.configuration import inject_dependencies

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


# Inject dependencies and configurations
inject_dependencies()
