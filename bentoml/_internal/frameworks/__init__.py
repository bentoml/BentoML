import re
import typing as t
from typing import TYPE_CHECKING

import attr

from ..types import Tag
from ..runner.runner import Runner
from ..runner.runner import BatchOptions
from ..runner.runner import ResourceQuota
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    import numpy as np

    from ..models import ModelStore

    AnyNdarray = np.ndarray[t.Any, np.dtype[t.Any]]


VARNAME_RE = re.compile(r"\W|^(?=\d)")


def _name_converter(name: str) -> str:
    if not name.isidentifier():
        return VARNAME_RE.sub("_", name)
    return name


def _batch_option_converter(
    d: t.Union[None, t.Dict[str, t.Any], BatchOptions]
) -> BatchOptions:
    if d is None:
        return BatchOptions()
    if isinstance(d, BatchOptions):
        return d
    return BatchOptions(**d)


def _quota_converter(
    d: t.Union[None, t.Dict[str, t.Any], ResourceQuota]
) -> ResourceQuota:
    if d is None:
        return ResourceQuota()
    if isinstance(d, ResourceQuota):
        return d
    return ResourceQuota(**d)


@attr.define(kw_only=True)
class ModelRunner(Runner):
    name: str = attr.ib(converter=_name_converter)
    batch_options: BatchOptions = attr.ib(converter=_batch_option_converter)
    resource_quota: ResourceQuota = attr.ib(converter=_quota_converter)

    model_store: "ModelStore" = attr.ib(factory=BentoMLContainer.model_store.get)
    tag: Tag = attr.ib()

    # no init attrs
    _model: t.Any = attr.ib(init=False)
    _predict_fn: t.Callable[..., t.Any] = attr.ib(init=False)

    @property
    def required_models(self) -> t.List[Tag]:
        return [self.tag]
