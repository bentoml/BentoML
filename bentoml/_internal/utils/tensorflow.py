import typing as t
import importlib.util
from typing import TYPE_CHECKING

from bentoml.exceptions import BentoMLException

from ..types import LazyType
from ..types import FrozenDict
from ..types import FrozenList
from .lazy_loader import LazyLoader

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

if TYPE_CHECKING:
    from ..external_typing import tensorflow as tf
else:
    tf = LazyLoader(
        "tf",
        globals(),
        "tensorflow",
        exc_msg="`tensorflow` is required to use bentoml.tensorflow module.",
    )

TF_KERAS_DEFAULT_FUNCTIONS = {
    "_default_save_signature",
    "call_and_return_all_conditional_losses",
}

TENSOR_CLASS_NAMES = (
    "RaggedTensor",
    "SparseTensor",
    "TensorArray",
    "EagerTensor",
    "Tensor",
)

__all__ = [
    "get_tf_version",
    "tf_function_wrapper",
    "pretty_format_restored_model",
    "is_gpu_available",
]


def is_gpu_available() -> bool:
    try:
        return len(tf.config.list_physical_devices("GPU")) > 0
    except AttributeError:
        return tf.test.is_gpu_available()


def get_tf_version() -> str:
    # courtesy of huggingface/transformers

    _tf_version = ""
    _tf_available = importlib.util.find_spec("tensorflow") is not None
    if _tf_available:
        candidates = (
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "intel-tensorflow-avx512",
            "tensorflow-rocm",
            "tensorflow-macos",
        )
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for pkg in candidates:
            try:
                _tf_version = importlib_metadata.version(pkg)
                break
            except importlib_metadata.PackageNotFoundError:
                pass
    return _tf_version


def isinstance_spec_wrapper(
    tensor: t.Union[
        "tf.CastableTensorType", "tf.TensorType", "tf.UnionTensorSpec", "tf.TypeSpec"
    ],
    tensor_spec: t.Union[str, t.Tuple[str, ...], "tf.TensorSpec"],
    class_name: t.Optional[str] = None,
) -> bool:
    """
    :code:`isinstance` wrapper to check for TensorSpec type.

    Args:
        tensor (`Union[tf.Tensor, tf.EagerTensor, tf.SparseTensor, tf.RaggedTensor]`):
            tensor class to check.
        tensor_spec (`Union[str, Tuple[str,...]]`):
            class used to check with :obj:`tensor`. Follows :obj:`TENSOR_CLASS_NAME`

    Returns:
        `bool` if given tensor match a given spec
    """
    if not tensor_spec:
        return False
    if isinstance(tensor_spec, str):
        return type(tensor).__name__ == tensor_spec.split(".")[-1]
    elif isinstance(tensor_spec, tuple):
        return any(isinstance_spec_wrapper(tensor, k) for k in tensor_spec)
    else:
        assert class_name is not None
        return LazyType["tf.TensorSpec"](class_name).isinstance(tensor)


def normalize_spec(value: t.Any) -> "tf.TypeSpec":
    """normalize tensor spec"""
    if not isinstance_spec_wrapper(value, TENSOR_CLASS_NAMES):
        return value
    if isinstance_spec_wrapper(value, "RaggedTensor", "tf.RaggedTensor"):
        return tf.RaggedTensorSpec.from_value(value)
    if isinstance_spec_wrapper(value, "SparseTensor", "tf.SparseTensor"):
        return tf.SparseTensorSpec.from_value(value)
    if isinstance_spec_wrapper(value, "TensorArray", "tf.TensorArray"):
        return tf.TensorArraySpec.from_value(value)
    if isinstance_spec_wrapper(value, ("Tensor", "EagerTensor")):
        return tf.TensorSpec.from_tensor(value)
    raise BentoMLException(f"Unknown type for tensor spec, got{type(value)}.")


def get_input_signatures(
    func: t.Union["tf.RestoredFunction", "tf.ConcreteFunction"]
) -> t.Union[t.Tuple[t.Union["tf.TensorSpec", t.Dict[t.Any, t.Any]]], t.Tuple[t.Any]]:
    if hasattr(func, "function_spec"):  # for RestoredFunction
        func_spec: "tf.FunctionSpec" = getattr(func, "function_spec")
        if hasattr(func_spec, "input_signature"):
            return ((getattr(func_spec, "input_signature"), {}),)
        else:
            concrete_func: t.List["tf.ConcreteFunction"] = getattr(
                func, "concrete_functions"
            )
            return tuple(
                s for conc in concrete_func for s in get_input_signatures(conc)
            )
    if hasattr(func, "structured_input_signature"):  # for ConcreteFunction
        if getattr(func, "structured_input_signature") is not None:
            return (getattr(func, "structured_input_signature"),)

        # TODO(bojiang): using private API
        if getattr(func, "_arg_keywords") is not None:
            return (
                (
                    tuple(),
                    {
                        k: normalize_spec(v)
                        for k, v in zip(getattr(func, "_arg_keywords"), func.inputs)  # type: ignore
                    },
                ),
            )
    return tuple()


def get_output_signature(
    func: t.Union["tf.RestoredFunction", "tf.ConcreteFunction"]
) -> t.Union["tf.ConcreteFunction", t.Tuple[t.Any, ...], t.Dict[str, "tf.TypeSpec"]]:
    if hasattr(func, "function_spec"):  # for RestoredFunction
        # assume all concrete functions have same signature
        concrete_function_wrapper: "tf.ConcreteFunction" = getattr(
            func, "concrete_functions"
        )[0]
        return get_output_signature(concrete_function_wrapper)

    if hasattr(func, "structured_input_signature"):  # for ConcreteFunction
        if getattr(func, "structured_outputs") is not None:
            outputs = getattr(func, "structured_outputs")
            if LazyType["t.Dict[str, tf.TensorSpec]"](dict).isinstance(outputs):
                return {k: normalize_spec(v) for k, v in outputs.items()}
            return outputs
        else:
            outputs: t.Tuple["tf.TensorSpec"] = getattr(func, "outputs")
            return tuple(normalize_spec(v) for v in outputs)

    return tuple()


def get_arg_names(
    func: t.Union["tf.RestoredFunction", "tf.ConcreteFunction"]
) -> t.List[str]:
    if hasattr(func, "function_spec"):  # for RestoredFunction
        func_spec: "tf.FunctionSpec" = getattr(func, "function_spec")
        return getattr(func_spec, "arg_names")
    if hasattr(func, "structured_input_signature"):  # for ConcreteFunction
        return getattr(func, "_arg_keywords")
    raise BentoMLException(
        f"given `func` is not either a RestoredFunction or ConcreteFunction, got type {type(func)} instead."
    )


def get_restored_functions(m: "tf.Trackable") -> t.Dict[str, "tf.RestoredFunction"]:
    function_map = {k: getattr(m, k) for k in dir(m)}
    restore_funcs: t.Dict[str, "tf.RestoredFunction"] = {}
    for k, v in function_map.items():
        if k not in TF_KERAS_DEFAULT_FUNCTIONS and hasattr(v, "function_spec"):
            restore_funcs[k] = v
    return restore_funcs


def get_serving_default_function(m: "tf.Trackable") -> "tf.ConcreteFunction":
    if not hasattr(m, "signatures"):
        raise EnvironmentError(f"{type(m)} is not a valid SavedModel format.")
    signatures: "tf.SignatureMap" = getattr(m, "signatures")
    func = signatures.get(tf.compat.v2.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY)  # type: ignore
    if func is not None:
        return func
    raise BentoMLException(
        "Given Trackable objects doesn't contain a default functions from SignatureMap"
    )


def _pretty_format_function_call(base: str, name: str, arg_names: t.Tuple[t.Any]):
    if arg_names:
        part_sigs = ", ".join(f"{k}" for k in arg_names)
    else:
        part_sigs = ""

    if name == "__call__":
        return f"{base}({part_sigs})"
    return f"{base}.{name}({part_sigs})"


def _pretty_format_positional(positional: t.Tuple["tf.TensorSpec"]) -> str:
    if len(positional) != 0:
        return f'Positional arguments ({len(positional)} total):\n    {"    * ".join(str(a) for a in positional)}'  # noqa
    return "No positional arguments.\n"


def pretty_format_function(
    function: t.Union["tf.RestoredFunction", "tf.ConcreteFunction"],
    obj: str = "<object>",
    name: str = "<function>",
) -> str:
    ret = ""
    outs = get_output_signature(function)
    sigs = get_input_signatures(function)
    arg_names = get_arg_names(function)

    if hasattr(function, "function_spec"):
        arg_names = getattr(function, "function_spec").arg_names
    else:
        arg_names = getattr(function, "_arg_keywords")

    ret += _pretty_format_function_call(obj, name, arg_names)
    ret += "\n------------\n"

    signature_descriptions = []  # type: t.List[str]

    for index, sig in enumerate(sigs):
        positional, keyword = sig  # type: ignore
        signature_descriptions.append(f"Arguments Option {index + 1}:\n  {_pretty_format_positional(positional)}\n  Keyword arguments:\n    {keyword}")  # type: ignore

    ret += "\n\n".join(signature_descriptions)
    ret += f"\n\nReturn:\n  {outs}\n\n"
    return ret


def pretty_format_restored_model(model: "tf.AutoTrackable") -> str:
    WARNING_MSG = """\
    No serving function was found in the saved model.
    In the model implementation, use `tf.function` decorator to mark the method needed for model serving. \n
    Find more details in related TensorFlow docs here
    https://www.tensorflow.org/api_docs/python/tf/saved_model/save
    """  # noqa
    part_functions = ""

    if get_tf_version().startswith("2"):
        restored_functions = get_restored_functions(model)
        for name, func in restored_functions.items():
            part_functions += pretty_format_function(func, "model", name)
            part_functions += "\n"
        if not restored_functions:
            return WARNING_MSG

    else:
        serving_default = get_serving_default_function(model)
        if serving_default:
            part_functions += pretty_format_function(
                serving_default, "model", "signatures['serving_default']"
            )
            part_functions += "\n"
        if not serving_default:
            return WARNING_MSG

    return f"Found restored functions:\n{part_functions}"


def cast_tensor_by_spec(
    _input: "tf.TensorType", spec: "tf.TypeSpec"
) -> "t.Union[tf.CastableTensorType, tf.TensorType]":
    """
    transform dtype & shape following spec
    """
    print(spec)
    if not isinstance_spec_wrapper(spec, "TensorSpec"):
        return _input

    if LazyType["tf.CastableTensorType"]("tf.Tensor").isinstance(_input):
        # TensorFlow issue#43038
        # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        return tf.cast(_input, dtype=spec.dtype, name=spec.name)  # type: ignore
    return tf.constant(_input, dtype=spec.dtype, name=spec.name)  # type: ignore


class tf_function_wrapper:  # pragma: no cover
    def __init__(
        self,
        origin_func: t.Callable[..., t.Any],
        arg_names: t.List[str] = FrozenList(),
        arg_specs: t.Tuple["tf.TensorSpec"] = tuple(),
        kwarg_specs: t.Dict[str, "tf.TensorSpec"] = FrozenDict(),
    ) -> None:
        self.origin_func = origin_func
        self.arg_names = arg_names
        self.arg_specs = arg_specs
        self.kwarg_specs = {k: v for k, v in zip(arg_names or [], arg_specs or [])}
        self.kwarg_specs.update(kwarg_specs or {})

    def __call__(self, *args: "tf.TensorType", **kwargs: "tf.TensorType") -> t.Any:
        if self.arg_specs is None and self.kwarg_specs is None:
            return self.origin_func(*args, **kwargs)

        for k in kwargs:
            if k not in self.kwarg_specs:
                raise TypeError(f"Function got an unexpected keyword argument {k}")

        arg_keys = {k for k, _ in zip(self.arg_names, args)}
        _ambiguous_keys = arg_keys & set(kwargs)  # type: t.Set[str]
        if _ambiguous_keys:
            raise TypeError(f"got two values for arguments '{_ambiguous_keys}'")

        # INFO:
        # how signature with kwargs works?
        # https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/python/eager/function.py#L1519
        transformed_args: t.Tuple[t.Any, ...] = tuple(
            cast_tensor_by_spec(arg, spec) for arg, spec in zip(args, self.arg_specs)
        )

        transformed_kwargs = {
            k: cast_tensor_by_spec(arg, self.kwarg_specs[k])
            for k, arg in kwargs.items()
        }
        return self.origin_func(*transformed_args, **transformed_kwargs)

    def __getattr__(self, k: t.Any) -> t.Any:
        return getattr(self.origin_func, k)

    @classmethod
    def hook_loaded_model(cls, loaded_model: t.Any) -> None:
        funcs = get_restored_functions(loaded_model)
        for k, func in funcs.items():
            arg_names = get_arg_names(func)
            sigs = get_input_signatures(func)
            if not sigs:
                continue
            arg_specs: "t.Tuple[tf.TensorSpec]" = sigs[0][0]  # type: ignore
            kwarg_specs: "t.Dict[str, tf.TensorSpec]" = sigs[0][1]  # type: ignore
            setattr(
                loaded_model,
                k,
                cls(
                    func,
                    arg_names=arg_names,
                    arg_specs=arg_specs,
                    kwarg_specs=kwarg_specs,
                ),
            )
