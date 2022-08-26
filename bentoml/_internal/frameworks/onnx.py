from __future__ import annotations

import os
import typing as t
import logging
from types import ModuleType
from typing import TYPE_CHECKING

import attr
import numpy as np

import bentoml
from bentoml import Tag
from bentoml.models import ModelContext
from bentoml.models import ModelOptions
from bentoml.exceptions import NotFound
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..types import LazyType
from ..utils.pkg import get_pkg_version
from ..runner.utils import Params

if TYPE_CHECKING:
    from bentoml.types import ModelSignature
    from bentoml.types import ModelSignatureDict


try:
    import onnx
except ImportError:  # pragma: no cover (trivial error checking)
    raise MissingDependencyException(
        "onnx is required in order to use module `bentoml.onnx`, install "
        "onnx with `pip install onnx`. For more information, refer to "
        "https://onnx.ai/get-started.html"
    )

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover (trivial error checking)
    raise MissingDependencyException(
        "`onnxruntime` or `onnxruntime-gpu` is required by `bentoml.onnx`, "
        "install onnxruntime or onnxruntime-gpu with `pip install onnxruntime` "
        "or `pip install onnxruntime-gpu`. For more information, refer to "
        "https://onnxruntime.ai/"
    )

MODULE_NAME = "bentoml.onnx"
MODEL_FILENAME = "saved_model.onnx"
API_VERSION = "v1"

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch  # noqa

    from .. import external_typing as ext
    from ..external_typing import tensorflow as tf_ext  # noqa

    ProvidersType = list[str | t.Tuple[str, t.Dict[str, t.Any]]]
    ONNXArgType = t.Union[
        "ext.NpNDArray", "ext.PdDataFrame", "torch.Tensor", "tf_ext.Tensor"
    ]


# helper methods
def _yield_providers(
    iterable: t.Sequence[t.Any],
) -> t.Generator[str, None, None]:  # pragma: no cover
    if isinstance(iterable, tuple):
        yield iterable[0]
    elif isinstance(iterable, str):
        yield iterable
    else:
        yield from iterable


def flatten_list(lst: t.List[t.Any]) -> t.List[str]:  # pragma: no cover
    return [k for i in lst for k in _yield_providers(i)]


@attr.define
class ONNXOptions(ModelOptions):
    """Options for the ONNX model"""

    providers: list[str] | None = attr.field(default=None)
    session_options: "ort.SessionOptions" | None = attr.field(default=None)


def get(tag_like: str | Tag) -> bentoml.Model:
    """
    Get the BentoML model with the given tag.

    Args:
        tag_like: The tag of the model to retrieve from the model store.

    Returns:
        :obj:`~bentoml.Model`: A BentoML :obj:`~bentoml.Model` with the matching tag.

    Example:

    .. code-block:: python

       import bentoml
       # target model must be from the BentoML model store
       model = bentoml.onnx.get("onnx_resnet50")
    """
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def load_model(
    bento_model: str | Tag | bentoml.Model,
    *,
    providers: ProvidersType | None = None,
    session_options: "ort.SessionOptions" | None = None,
) -> ort.InferenceSession:
    """
    Load the onnx model with the given tag from the local BentoML model store.

    Args:
        bento_model (``str`` ``|`` :obj:`~bentoml.Tag` ``|`` :obj:`~bentoml.Model`):
            Either the tag of the model to get from the store, or a BentoML `~bentoml.Model`
            instance to load the model from.
        providers (`List[Union[str, Tuple[str, Dict[str, Any]]`, `optional`, default to :code:`None`):
            Different providers provided by users. By default BentoML will use ``["CPUExecutionProvider"]``
            when loading a model.
        session_options (`onnxruntime.SessionOptions`, `optional`, default to :code:`None`):
            SessionOptions per use case. If not specified, then default to :code:`None`.

    Returns:
        :obj:`onnxruntime.InferenceSession`:
            An instance of ONNX Runtime inference session created using ONNX model loaded from the model store.

    Example:

    .. code-block:: python

        import bentoml
        sess = bentoml.onnx.load_model("my_onnx_model")
    """  # noqa

    if not isinstance(bento_model, bentoml.Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, not loading with {MODULE_NAME}."
        )

    if providers:
        if not all(i in ort.get_all_providers() for i in flatten_list(providers)):
            raise BentoMLException(f"'{providers}' cannot be parsed by `onnxruntime`")
    else:
        providers = ["CPUExecutionProvider"]

    return ort.InferenceSession(
        bento_model.path_of(MODEL_FILENAME),
        sess_options=session_options,
        providers=providers,
    )


def save_model(
    name: str,
    model: onnx.ModelProto,
    *,
    signatures: dict[str, ModelSignatureDict | ModelSignature] | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """Save a onnx model instance to the BentoML model store.

    Args:
        name (``str``):
            The name to give to the model in the BentoML store. This must be a valid
            :obj:`~bentoml.Tag` name.
        model (:obj:`~onnx.ModelProto`):
            The ONNX model to be saved.
        signatures (``dict[str, ModelSignatureDict]``, optional):
            Signatures of predict methods to be used. If not provided, the signatures default to
            ``{"run": {"batchable": False}}``. Because we are using :obj:``onnxruntime.InferenceSession``,
            the only allowed method name is "run"
            See :obj:`~bentoml.types.ModelSignature` for more details.
        labels (``dict[str, str]``, optional):
            A default set of management labels to be associated with the model. An example is
            ``{"training-set": "data-1"}``.
        custom_objects (``dict[str, Any]``, optional):
            Custom objects to be saved with the model. An example is
            ``{"my-normalizer": normalizer}``.

            Custom objects are currently serialized with cloudpickle, but this implementation is
            subject to change.
        external_modules (:code:`List[ModuleType]`, `optional`, default to :code:`None`):
            user-defined additional python modules to be saved alongside the model or custom objects,
            e.g. a tokenizer module, preprocessor module, model configuration module
        metadata (``dict[str, Any]``, optional):
            Metadata to be associated with the model. An example is ``{"bias": 4}``.

            Metadata is intended for display in a model management UI and therefore must be a
            default Python type, such as ``str`` or ``int``.

    Returns:

        :obj:`~bentoml.Model`: A BentoML model containing the saved
        ONNX model instance.  store.

    Example:

    .. code-block:: python

        import bentoml

        import torch
        import torch.nn as nn

        class ExtendedModel(nn.Module):
            def __init__(self, D_in, H, D_out):
                # In the constructor we instantiate two nn.Linear modules and assign them as
                #  member variables.
                super(ExtendedModel, self).__init__()
                self.linear1 = nn.Linear(D_in, H)
                self.linear2 = nn.Linear(H, D_out)

            def forward(self, x, bias):
                # In the forward function we accept a Tensor of input data and an optional bias
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred + bias


        N, D_in, H, D_out = 64, 1000, 100, 1
        x = torch.randn(N, D_in)
        model = ExtendedModel(D_in, H, D_out)

        input_names = ["x", "bias"]
        output_names = ["output1"]

        tmpdir = "/tmp/model"
        model_path = os.path.join(tmpdir, "test_torch.onnx")
        torch.onnx.export(
            model,
            (x, torch.Tensor([1.0])),
            model_path,
            input_names=input_names,
            output_names=output_names,
        )

        bento_model = bentoml.onnx.save_model("onnx_model", model_path, signatures={"run": "batchable": True})

    """

    try:
        import importlib.metadata as importlib_metadata
    except ImportError:
        import importlib_metadata

    # prefer "onnxruntime-gpu" over "onnxruntime" for framework versioning
    _PACKAGE = ["onnxruntime-gpu", "onnxruntime", "onnxruntime-silicon"]
    for p in _PACKAGE:
        try:
            _onnxruntime_version = get_pkg_version(p)
            _onnxruntime_pkg = p
            break
        except importlib_metadata.PackageNotFoundError:
            pass

    context = ModelContext(
        framework_name="onnx",
        framework_versions={
            "onnx": get_pkg_version("onnx"),
            _onnxruntime_pkg: _onnxruntime_version,
        },
    )

    if signatures is None:
        signatures = {
            "run": {"batchable": False},
        }
        logger.info(
            f"Using the default model signature for onnx ({signatures}) for model {name}."
        )
    else:
        provided_methods = list(signatures.keys())
        if provided_methods != ["run"]:
            raise BentoMLException(
                "bentoml.onnx will load onnx model into an "
                "`onnxruntime.InferenceSession` for inference, hence the only "
                f"allowed method name is `run` instead of provided {provided_methods}"
            )

    if not isinstance(model, onnx.ModelProto):
        raise TypeError(f"Given model ({model}) is not a onnx.ModelProto.")

    if len(model.graph.output) > 1:
        logger.warning(
            "The model you are attempting to save has more than one output. The ONNX runner will only return the first output."
        )

    options = ONNXOptions()

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        signatures=signatures,
        labels=labels,
        options=options,
        custom_objects=custom_objects,
        external_modules=external_modules,
        metadata=metadata,
        context=context,
    ) as bento_model:
        onnx.save(model, bento_model.path_of(MODEL_FILENAME))

        return bento_model


def get_runnable(bento_model: bentoml.Model) -> t.Type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    class ONNXRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()

            session_options = (
                bento_model.info.options.session_options or ort.SessionOptions()
            )

            # check for resources
            available_gpus = os.getenv("CUDA_VISIBLE_DEVICES")
            if available_gpus is not None and available_gpus not in ("", "-1"):
                # assign GPU resources
                providers = bento_model.info.options.providers or [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]

            else:
                # assign CPU resources

                # If onnxruntime-gpu is installed,
                # CUDAExecutionProvider etc. will be available even no
                # GPU is presented in system, which may result some
                # error when initializing ort.InferenceSession
                providers = bento_model.info.options.providers or [
                    "CPUExecutionProvider"
                ]

                # set CPUExecutionProvider parallelization options
                # TODO @larme: follow onnxruntime issue 11668 and
                # 10330 to decide best cpu parallelization strategy
                thread_count = int(os.getenv("BENTOML_NUM_THREAD", 1))
                session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                if session_options.intra_op_num_threads != 0:
                    logger.warning(
                        "You have modified default setting of "
                        "onnxruntime.InferenceSession's `intra_op_num_threads` "
                        "setting, bentoml.onnx will try to manage CPU resource "
                        "for you and override this setting"
                    )
                session_options.intra_op_num_threads = thread_count
                if session_options.inter_op_num_threads != 0:
                    logger.warning(
                        "You have modified default setting of "
                        "onnxruntime.InferenceSession's `inter_op_num_threads` "
                        "setting, bentoml.onnx will try to manage CPU resource "
                        "for you and override this setting"
                    )
                session_options.inter_op_num_threads = thread_count

            self.model = load_model(
                bento_model, session_options=session_options, providers=providers
            )

            self.predict_fns: dict[str, t.Callable[..., t.Any]] = {}
            for method_name in bento_model.info.signatures:
                self.predict_fns[method_name] = getattr(self.model, method_name)

    def _mapping(item: ONNXArgType) -> "ext.NpNDArray":

        # currently ort only support np.float32
        # https://onnxruntime.ai/docs/api/python/auto_examples/plot_common_errors.html
        # TODO @larme: what dtype of input to use if we use FP16? ref: onnxruntime issue 11768
        if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(item):
            if item.dtype != np.float32:
                item = item.astype(np.float32, copy=False)
        elif LazyType["ext.PdDataFrame"]("pandas.DataFrame").isinstance(item):
            item = item.to_numpy(dtype=np.float32)
        elif LazyType["tf.Tensor"]("tensorflow.Tensor").isinstance(item):
            item = np.array(memoryview(item)).astype(np.float32, copy=False)  # type: ignore
        elif LazyType["torch.Tensor"]("torch.Tensor").isinstance(item):
            item = item.numpy().astype(np.float32, copy=False)
        else:
            raise TypeError(
                "`run` of ONNXRunnable only takes "
                "`numpy.ndarray` or `pd.DataFrame`, `tf.Tensor`, or `torch.Tensor` as input parameters"
            )
        return item

    def add_runnable_method(method_name: str, options: ModelSignature):
        def _run(self, *args: ONNXArgType) -> t.Any:
            params = Params["ONNXArgType"](*args)
            params = params.map(_mapping)

            input_names = {
                i.name: val for i, val in zip(self.model.get_inputs(), params.args)
            }
            output_names = [o.name for o in self.model.get_outputs()]
            return self.predict_fns[method_name](output_names, input_names)[0]

        ONNXRunnable.add_method(
            _run,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in bento_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return ONNXRunnable
