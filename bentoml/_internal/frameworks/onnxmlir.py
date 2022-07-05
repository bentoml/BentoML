from __future__ import annotations

import os
import sys
import shutil
import typing as t
import logging
import subprocess
from copy import deepcopy
from uuid import uuid4
from typing import Iterable
from typing import TYPE_CHECKING
from pathlib import Path
from contextlib import contextmanager

import attr

import bentoml
from bentoml import Tag
from bentoml.models import ModelContext
from bentoml.models import ModelOptions
from bentoml.exceptions import NotFound
from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..utils import cached_contextmanager
from ..utils import resolve_user_filepath
from ..utils.pkg import find_spec
from ..utils.buildx import run_docker_cmd
from ..utils.buildx import requires_docker
from ..utils.buildx import UNAME_M_TO_PLATFORM_MAPPING

logger = logging.getLogger(__name__)

MODULE_NAME = "bentoml.onnxmlir"
MODEL_FILENAME = "saved_model.so"
API_VERSION = "v1"

# TODO: mount session to docker container and use prebuilt?
DEV_MLIR_IMAGE = "onnxmlirczar/onnx-mlir-dev:latest"

# release docker images to compile the binary
RELEASE_MLIR_IMAGE = "onnxmlirczar/onnx-mlir:latest"
DOCKER_WORKDIR = "/workdir"
DOCKER_OUTDIR = "/output"
COMPILE_ARG = ["--O3", "--EmitLib"]

# onnx-mlir constant
if sys.platform == "win32":
    _onnx_mlir_binary = "onnx-mlir.exe"
else:
    _onnx_mlir_binary = "onnx-mlir"

# There are loosely a few defined environment variables
# from onnx/onnx-mlir that are available throughout its codebase.
# here are for documentation
ACCEPTED_ONNX_ENVVAR = ["ONNX_MLIR_HOME", "ONNX_MLIR_BIN"]
USE_DOCKER_ENVVAR = "USE_ONNXMLIR_DOCKER"
use_docker = os.environ.get(USE_DOCKER_ENVVAR, str(False)).lower() == "true"

warning = """\
The compiled Python runtime 'PyRuntime' is required in order to use module `bentoml.onnxmlir`. Please set either ONNX_MLIR_HOME or ONNX_MLIR_BIN so that BentoML can find onnx-mlir binary and runtime library correctly:

1. Set ONNX_MLIR_HOME to the path of 'build' directory of onnx-mlir. The 'build' directory refers to parent folder containing 'bin', 'lib' in which ONNX-MLIR executables and libraries can be found, typically '/path/to/onnx-mlir/build/Debug'.
2. Set ONNX_MLIR_BIN to 'onnx-mlir/build/Debug/bin' directory. After, add $ONNX_MLIR_BIN to PATH environment variable: "export PATH=$ONNX_MLIR_BIN:$PATH" and reload current shell.
"""

if TYPE_CHECKING:
    from .. import external_typing as ext
    from ..types import PathType
    from ...types import ModelSignature
    from ..models.model import ModelSignaturesType

    class ExecutionSession:
        def __init__(self, shared_lib_path: str, use_default_entry_point: bool = ...):
            ...

        def run(self, input: list[ext.NpNDArray]) -> list[ext.NpNDArray]:
            ...

        def input_signature(self) -> str:
            ...

        def output_signature(self) -> str:
            ...

        def entry_points(self) -> list[str]:
            ...

        def set_entry_point(self, name: str):
            ...


@attr.define
class ONNXMLirOptions(ModelOptions):
    use_default_entry_point: bool = attr.field(default=True)
    use_docker: bool = attr.field(default=False)


def get(tag_like: str | Tag) -> bentoml.Model:
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def check_environment() -> tuple[str, str]:
    if use_docker:
        logger.debug(
            f"Using {DEV_MLIR_IMAGE} for prebuilt binary. It might take a while to pull if {DEV_MLIR_IMAGE} is not available locally."
        )
        # _onnx_mlir depends on changing cwd for subprocess
        return f"./{_onnx_mlir_binary}", f"{DOCKER_WORKDIR}/onnx-mlir/build/Debug/lib"
    else:
        _onnx_mlir, _onnx_mlir_runtime = "", ""
        _runtime_spec = find_spec("PyRuntime")
        if _runtime_spec and _runtime_spec.origin:
            # best effort to find onnx-mlir binary
            logger.debug(
                "BentoML will provide best effort to find onnx-mlir runtime. This behaviour depends on the output 'build' directory of onnx-mlir are kept as default. If you customized the output directory, you need to set either ONNX_MLIR_HOME or ONNX_MLIR_BIN manually."
            )
            pyruntime_binary = os.path.realpath(_runtime_spec.origin)
            _onnx_mlir_runtime = os.path.dirname(pyruntime_binary)
            _onnx_mlir = os.path.join(
                os.path.dirname(_onnx_mlir_runtime), "bin", _onnx_mlir_binary
            )
        elif "ONNX_MLIR_HOME" in os.environ:
            _onnx_mlir_runtime = os.path.join(os.environ["ONNX_MLIR_HOME"], "lib")
            _onnx_mlir = os.path.join(
                os.environ["ONNX_MLIR_HOME"], "bin", _onnx_mlir_binary
            )
        elif "ONNX_MLIR_BIN" in os.environ:
            _onnx_mlir_runtime = os.path.join(
                os.path.dirname(os.environ["ONNX_MLIR_BIN"]), "lib"
            )
            _onnx_mlir = os.path.join(os.environ["ONNX_MLIR_BIN"], _onnx_mlir_binary)
        return _onnx_mlir, _onnx_mlir_runtime


@contextmanager
@requires_docker
def run_onnxmlir_docker_cmd(
    bento_model: bentoml.Model,
    model_path: str | None = None,
    additional_cmd: list[str] | None = None,
    *,
    docker_platform: t.Literal["x86_64", "ppc64le", "s390x"] = "x86_64",
    _mlir_image_name: str = RELEASE_MLIR_IMAGE,
    **kwargs: t.Any,
) -> t.Generator[subprocess.Popen[str], None, None]:
    container_name = uuid4().hex
    _docker_binary: str = kwargs.pop("_docker_binary")
    target_path = bento_model.path_of("/")
    run_cmd = [
        _docker_binary,
        "run",
        "--rm",
        "-it",
        "--platform",
        UNAME_M_TO_PLATFORM_MAPPING[docker_platform],
        "--name",
        container_name,
    ]
    if model_path:
        model_dir = os.path.dirname(model_path)
        run_cmd.extend(["-v", f"{model_dir}:{DOCKER_WORKDIR}{model_dir}"])
    run_cmd.extend(
        [
            "-v",
            f"{target_path}:{DOCKER_OUTDIR}{target_path}",
            "-u",
            f"{os.geteuid()}:{os.getegid()}",
            _mlir_image_name,
            *(additional_cmd or []),
            *COMPILE_ARG,
            f"{DOCKER_WORKDIR}{model_path}",
            "-o",
            f"{DOCKER_OUTDIR}{bento_model.path_of(MODEL_FILENAME).split('.')[0]}",
        ]
    )
    with subprocess.Popen(
        run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs
    ) as proc:
        try:
            yield proc
        finally:
            subprocess.call([_docker_binary, "stop", container_name])


def load_model(bento_model: str | Tag | bentoml.Model) -> ExecutionSession:
    """
    Load ONNXMLir compiled model with the given tag from the local BentoML model store.

    Args:
        bento_model: Either the tag of the model to get from the store, or a BentoML `~bentoml.Model` instance to load the model from.
        use_default_entry_point: use the default entry point that is :code:`run_main_graph` or not. Set to :code:`True` by default.

    Returns:
        :obj:`ExecutionSession`: that can be used to run inference, that is loaded from the model store or BentoML :obj:`~bentoml.Model`.

    Example:
    .. code-block:: python

        from __future__ import annotations

        from typing import TYPE_CHECKING

        import bentoml
        import numpy as np

        if TYPE_CHECKING:
            from bentoml.onnxmlir import ExecutionSession

        session: ExecutionSession = bentoml.onnxmlir.load_model("compiled_mnist:latest")
        inputs = np.full((1, 1, 28, 28), 1, np.dtype(np.float32))
        session.run([inputs])
    """  # noqa
    if not isinstance(bento_model, bentoml.Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, not loading with {MODULE_NAME}."
        )
    use_default_entrypoint: bool = (
        bento_model.info.options.use_default_entry_point or use_default_entry_point
    )
    original_sys_path = deepcopy(sys.path)
    _, runtime = check_environment()
    sys.path.append(runtime)
    use_docker: bool = bento_model.info.options.use_docker
    if use_docker:
        platform = "x86_64"
        with run_onnxmlir_docker_cmd(
            bento_model,
            additional_cmd=["python"],
            _mlir_image_name=DEV_MLIR_IMAGE,
            docker_platform=platform,
        ) as proc:
            print(sys.path, sys.executable)
            print(proc)
            if not TYPE_CHECKING:
                from PyRuntime import ExecutionSession
    else:
        try:
            if not TYPE_CHECKING:
                from PyRuntime import ExecutionSession
        except ImportError:
            raise MissingDependencyException(warning)
        finally:
            # this is safer to do than pop the last element.
            sys.path = original_sys_path

    return ExecutionSession(
        shared_lib_path=bento_model.path_of(MODEL_FILENAME),
        use_default_entry_point=use_default_entrypoint,
    )


def _get_onnxmlir_version() -> str:  # pragma: no cover (trivial)
    import requests

    version_url = "https://raw.githubusercontent.com/onnx/onnx-mlir/main/VERSION_NUMBER"
    return requests.get(version_url).text.strip()


def save_model(
    name: str,
    model_path: Path | PathType,
    *,
    use_default_entry_point: bool = True,
    trust_compiled_model: bool = False,
    docker_platform: t.Literal["x86_64", "ppc64le", "s390x"] = "x86_64",
    signatures: ModelSignaturesType | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """
    Save a ONNX model, that is compiled by BentoML with ONNXMLir, to the BentoML model store.

    Args:
        name: The name to give to the model in the BentoML store. This must be a valid :obj:`~bentoml.Tag` name.
        model_path: The compiled model to be save.
        trust_compiled_model: If set to :code:`True`, a model path '.so' will be saved in the model store. By default this is set to :code:`False`.
                              If you have a compiled model, set `trust_compiled_model=True`
        docker_platform: supported docker platform for onnxmlirczar/onnx-mlir:latest. Currently only linux/amd64, linux/ppc64le, and linux/s390x are supported.
        signatures: Signatures of predict methods to be used. If not provided, the signatures default to
                    {"run": {"batchable": False}}. See :obj:`~bentoml.types.ModelSignature` for more details.
        labels: A default set of management labels to be associated with the model. An example is
                ``{"training-set": "data-1"}``.
        custom_objects: Custom objects to be saved with the model. An example is ``{"my-normalizer": normalizer}``.
                        Custom objects are currently serialized with cloudpickle, but this implementation is subject to change.
        metadata: Metadata to be associated with the model. An example is ``{"bias": 4}``.
                  Metadata is intended for display in a model management UI and therefore must be a
                  default Python type, such as ``str`` or ``int``.

    Returns:
        :obj:`~bentoml.Tag`: A tag that can be used to access the saved model from the BentoML model store.

    Example:

    .. code-block:: python

       import bentoml

       model = bentoml.onnxmlir.save_model("mlir_mnist", "/path/to/mnist.onnx")

    If you compile the model yourself, you can pass in :code:`trust_compiled_model=True` to save the compiled model directly:

    .. code-block:: python

       import bentoml

       model = bentoml.onnxmlir.save_model("mlir_mnist", "./mnist.so", trust_compiled_model=True)

    To use other platform, you can pass in :code:`docker_platform`:

    .. code-block:: python

       # compile model on s390x platform
       import bentoml

       model = bentoml.onnxmlir.save_model("mlir_mnist", "./mnist.onnx", docker_platform="s390x")
    """
    options = ONNXMLirOptions(
        use_default_entry_point=use_default_entry_point, use_docker=use_docker
    )
    print(options)

    context = ModelContext(
        framework_name="onnxmlir",
        framework_versions={"onnxmlir": _get_onnxmlir_version()},
    )

    if signatures is None:
        signatures = {
            "run": {"batchable": False},
        }
        logger.info(
            f"Using the default model signature for onnxmlir ({signatures}) for model {name}."
        )
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    if model_path.suffix != ".onnx":
        if model_path.suffix == ".so":
            if not trust_compiled_model:
                raise BentoMLException(
                    "'trust_compiled_model' must set to True when saving a compiled model."
                )
        else:
            raise BentoMLException("given 'model_path' has to be a ONNX model format.")

    model_path = resolve_user_filepath(model_path.__fspath__(), ctx=None)

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        signatures=signatures,
        labels=labels,
        custom_objects=custom_objects,
        options=options,
        metadata=metadata,
        context=context,
    ) as bento_model:
        if trust_compiled_model and model_path.endswith(".so"):
            shutil.copyfile(model_path, bento_model.path_of(MODEL_FILENAME))
            return bento_model
        else:
            try:
                if use_docker:
                    # docker run --rm -it -v $(pwd):/workdir onnxmlirczar/onnx-mlir:latest -O3 --EmitLib /workdir/docs/mnist_example/mnist.onnx -o torch_mnist
                    logger.info(
                        f"Using {RELEASE_MLIR_IMAGE} to compile given model {model_path}. It might take a while if this is the first time you download {RELEASE_MLIR_IMAGE}."
                    )
                    container_name = f"onnxmlir-compiled-{uuid4()}"
                    # if platform.machine() not in ["x86_64", "ppc64le", "s390x"]:
                    #     raise BentoMLException(
                    #         f"onnxmlirczar/onnx-mlir doesn't contain a image for {platform.machine()}.\n{warning}\nRefers to https://github.com/onnx/onnx-mlir for more information."
                    #     )
                    model_dir = os.path.dirname(model_path)
                    target_path = bento_model.path_of("/")
                    DOCKER_CMD = [
                        "run",
                        "--rm",
                        "-it",
                        "--name",
                        container_name,
                        "--platform",
                        UNAME_M_TO_PLATFORM_MAPPING[docker_platform],
                        "-v",
                        f"{model_dir}:{DOCKER_WORKDIR}{model_dir}",
                        "-v",
                        f"{target_path}:{DOCKER_OUTDIR}{target_path}",
                        "-u",
                        f"{os.geteuid()}:{os.getegid()}",
                        RELEASE_MLIR_IMAGE,
                        *COMPILE_ARG,
                        f"{DOCKER_WORKDIR}{model_path}",
                        "-o",
                        f"{DOCKER_OUTDIR}{bento_model.path_of(MODEL_FILENAME).split('.')[0]}",
                    ]
                    run_docker_cmd(DOCKER_CMD)
                else:
                    binary, _ = check_environment()

                    subprocess.check_output(
                        [
                            binary,
                            *COMPILE_ARG,
                            model_path,
                            "-o",
                            bento_model.path_of(MODEL_FILENAME.split(".")[0]),
                        ]
                    )
            except Exception:
                raise
            else:
                return bento_model


def get_runnable(bento_model: bentoml.Model) -> t.Type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    class ONNXMLirRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("cpu",)  # TODO: support other accelerators
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()

            self.model = load_model(bento_model)

            self.entrypoints: dict[str, t.Callable[..., t.Any]] = {}
            for method_name in bento_model.info.signatures:
                try:
                    self.entrypoints[method_name] = getattr(self.model, method_name)
                except AttributeError:
                    raise InvalidArgument(
                        f"No method with name {method_name} found for compiled model {self.model}"
                    )

    def add_runnable_method(method_name: str, options: ModelSignature):
        def _run(
            self: ONNXMLirRunnable, input_data: list[ext.NpNDArray] | ext.NpNDArray
        ) -> list[ext.NpNDArray]:
            if not isinstance(input_data, Iterable):
                input_data = [input_data]
            return self.entrypoints[method_name](input=input_data)

        ONNXMLirRunnable.add_method(
            _run,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in bento_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return ONNXMLirRunnable
