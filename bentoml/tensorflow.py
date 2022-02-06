from ._internal.utils.tensorflow import get_tf_version as _get_tf_version

if _get_tf_version().startswith("2"):
    from ._internal.frameworks.tensorflow_v2 import load
    from ._internal.frameworks.tensorflow_v2 import save
    from ._internal.frameworks.tensorflow_v2 import load_runner
    from ._internal.frameworks.tensorflow_v2 import import_from_tfhub
else:
    from ._internal.frameworks.tensorflow_v1 import load
    from ._internal.frameworks.tensorflow_v1 import save
    from ._internal.frameworks.tensorflow_v1 import load_runner
    from ._internal.frameworks.tensorflow_v1 import import_from_tfhub

__all__ = ["load", "load_runner", "save", "import_from_tfhub"]

del _get_tf_version
