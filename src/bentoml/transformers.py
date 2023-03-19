from __future__ import annotations

import typing as _t
import logging as _logging

import attr as _attr

from ._internal.utils import LazyLoader as _LazyLoader
from ._internal.runner.runnable import Runnable as _Runnable
from ._internal.frameworks.transformers import get
from ._internal.frameworks.transformers import load_model
from ._internal.frameworks.transformers import save_model
from ._internal.frameworks.transformers import get_runnable
from ._internal.frameworks.transformers import import_pretrained
from ._internal.frameworks.transformers import TransformersOptions as ModelOptions

if _t.TYPE_CHECKING:
    import transformers

    from ._internal.tag import Tag
    from ._internal.models import Model
else:
    transformers = _LazyLoader("transformers", globals(), "transformers")
    del _LazyLoader


_logger = _logging.getLogger(__name__)


def save(tag: str, *args: _t.Any, **kwargs: _t.Any):
    _logger.warning(
        'The "%s.save" method is deprecated. Use "%s.save_model" instead.',
        __name__,
        __name__,
    )
    return save_model(tag, *args, **kwargs)


def load(tag: Tag | str, *args: _t.Any, **kwargs: _t.Any):
    _logger.warning(
        'The "%s.load" method is deprecated. Use "%s.load_model" instead.',
        __name__,
        __name__,
    )
    return load_model(tag, *args, **kwargs)


def load_runner(tag: Tag | str, *args: _t.Any, **kwargs: _t.Any):
    if len(args) != 0 or len(kwargs) != 0:
        _logger.error(
            'The "%s.load_runner" method is deprecated. "load_runner" arguments will be ignored. Use "%s.get("%s").to_runner()" instead.',
            __name__,
            __name__,
            tag,
        )
    else:
        _logger.warning(
            'The "%s.load_runner" method is deprecated. Use "%s.get("%s").to_runner()" instead.',
            __name__,
            __name__,
            tag,
        )
    return get(tag).to_runner()


_object_setattr = object.__setattr__


@_attr.define(slots=False)
class _PreTrainedHolder:
    name: str
    _options: ModelOptions

    def __init__(self, name: str, ref: Model):
        self.__attrs_init__(name=name, options=ref.info.options)  # type: ignore (attrs protocol)
        for pretrained_ref, klass_name in _t.cast(
            ModelOptions, ref.info.options
        ).pretrained.items():
            _object_setattr(
                self,
                pretrained_ref,
                getattr(transformers, klass_name).from_pretrained(
                    ref.path_of(pretrained_ref)
                ),
            )


class PreTrainedRunnable(_Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    _cached_ref_mapping: dict[str, Model]

    @staticmethod
    def _normalized_name(name: str) -> str:
        return name.replace("-", "_")

    def __init_subclass__(cls, *, models: list[str] | None = None):
        _cached_ref_mapping: dict[str, Model] = {}
        if models is not None:
            for model in models:
                ref = get(model)
                options = _t.cast(ModelOptions, ref.info.options)
                if len(options.pretrained) == 0:
                    raise ValueError(
                        f"Model '{model}' is not a pretrained model (not saved with 'import_pretrained')."
                    )
                _cached_ref_mapping[cls._normalized_name(ref.tag.name)] = ref
        cls._cached_ref_mapping = _cached_ref_mapping

    def __init__(self):
        for name, ref in self._cached_ref_mapping.items():
            if hasattr(self, name):
                _logger.warning("Overriding existing attribute '%s'.", name)
            _object_setattr(self, name, _PreTrainedHolder(name, ref))

    def from_pretrained(self, name: str):
        for normalized, ref in self._cached_ref_mapping.items():
            if hasattr(self, normalized):
                # NOTE: this is default __init__
                return getattr(getattr(self, normalized), name)
            options = _t.cast(ModelOptions, ref.info.options)
            if name in options.pretrained:
                return getattr(transformers, options.pretrained[name]).from_pretrained(
                    ref.path_of(name)
                )
        raise ValueError(
            f"Attribute '{name}' is not available in any of the models (available=[{', '.join(self._cached_ref_mapping.keys())}])."
        )


__all__ = [
    "load_model",
    "save_model",
    "import_pretrained",
    "get",
    "get_runnable",
    "ModelOptions",
    "PreTrainedRunnable",
]
