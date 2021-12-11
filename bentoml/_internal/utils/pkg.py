try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

get_pkg_version = importlib_metadata.version
