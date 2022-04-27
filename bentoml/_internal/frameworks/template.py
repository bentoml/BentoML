from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import bentoml
from bentoml import __version__ as BENTOML_VERSION
from bentoml.models import ModelOptions

if TYPE_CHECKING:
    ModelType = ...

MODULE_NAME = "bentoml.MY_MODULE"


@attr.define(frozen=True)
class FrameworkOptions(ModelOptions):
    pass


def load_model(
    bento_model: str | Tag | bentoml.Model,
    *,
    ...
) -> FrameworkModelType:
    """
    Load the <FRAMEWORK> model with the given tag from the local BentoML model store.

    Args:
        bento_model (``str`` ``|`` :obj:`~bentoml.Tag` ``|`` :obj:`~bentoml.Model`):
            Either the tag of the model to get from the store, or a BentoML `~bentoml.Model`
            instance to load the model from.
        ...
    Returns:
        <MODELTYPE>:
            The <FRAMEWORK> model loaded from the model store or BentoML :obj:`~bentoml.Model`.
    Example:
    .. code-block:: python
        import bentoml
        <LOAD EXAMPLE>
    """  # noqa
    if not isinstance(bento_model, bentoml.Model):
        bento_model = bentoml.models.get(bento_model)

    ...


def save_model(
    name: str,
    model: FrameworkModelType,
    *,
    signatures: t.Optional[t.Dict[str, t.Any]],
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    ...
) -> Tag:
    """
    Save a <FRAMEWORK> model instance to the BentoML model store.

    Args:
        name (``str``):
            The name to give to the model in the BentoML store. This must be a valid
            :obj:`~bentoml.Tag` name.
        model (<MODELTYPE>):
            The <FRAMEWORK> model to be saved.
    Returns:
        :obj:`~bentoml.Tag`: A tag that can be used to access the saved model from the BentoML model store.
    Example:
    .. code-block:: python
        <SAVE EXAMPLE>
    """  # noqa: LN001
    context = ModelContext(
        bentoml_version=BENTOML_VERSION,
        framework_name="<FRAMEWORK_PY>",
        framework_version={"<FRAMEWORK_PY>": <FRAMEWORK_PY>.__version__},
    )

    if signatures is None:
        signatures = {
            <DEFAULT_MODEL_METHOD>: {"batchable": False},
        }

    options = FrameworkOptions()

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        signatures=signatures,
        labels=labels,
        options=options,
        cusotm_objects=custom_objects,
        metadata=metadata,
        context=context,
    ) as bentoml_model:
        ...

        return _model.tag


def _to_runnable(
    bento_model: bentoml.Model,
) -> t.Type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    class FrameworkRunnable(bentoml.runner.Runnable):
        SUPPORT_NVIDIA_GPU = True
        SUPPORT_MULTIPLE_CPU_THREADS = True

        def __init__(self):
            super().__init__(**kwargs)
            # check for resources
            self.model = bento_model.load_model(bento_model.info.options.*)

    for method_name, options in bento_model.info.signatures.items():

        def _run(input_data) -> output_data:
            ...

        FrameworkRunnable.add_method(
            _run,
            name=method_name,
            batchable=True,
            batch_dim=...,
            input_spec=...,
            output_spec=...,
        )
