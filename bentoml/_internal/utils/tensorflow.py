import typing as t
import importlib.util
from typing import TYPE_CHECKING

from bentoml.exceptions import BentoMLException

from ..types import LazyType
from .lazy_loader import LazyLoader

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

if TYPE_CHECKING:
    import tensorflow as tf

    from ..external_typing import tensorflow as tf_ext
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


def check_tensor_spec(
    tensor: "tf_ext.TensorLike",
    tensor_spec: t.Union[str, t.Tuple[str, ...], t.List[str], "tf_ext.UnionTensorSpec"],
    class_name: t.Optional[str] = None,
) -> bool:
    """
    :code:`isinstance` wrapper to check spec for a given tensor.

    Args:
        tensor (`Union[tf.Tensor, tf.EagerTensor, tf.SparseTensor, tf.RaggedTensor]`):
            tensor class to check.
        tensor_spec (`Union[str, Tuple[str,...]]`):
            class used to check with :obj:`tensor`. Follows :obj:`TENSOR_CLASS_NAME`

    Returns:
        `bool` if given tensor match a given spec.
    """
    if tensor_spec is None:
        raise BentoMLException("`tensor` should not be None")
    tensor_cls = type(tensor).__name__
    if isinstance(tensor_spec, str):
        return tensor_cls == tensor_spec.split(".")[-1]
    elif isinstance(tensor_spec, (list, tuple, set)):
        return all(check_tensor_spec(tensor, k) for k in tensor_spec)
    else:
        if class_name is None:
            class_name = (
                str(tensor_spec.__class__).replace("<class '", "").replace("'>", "")
            )
        return LazyType["tf_ext.TensorSpec"](class_name).isinstance(tensor)


def normalize_spec(value: t.Any) -> "tf_ext.TypeSpec":
    """normalize tensor spec"""
    if not check_tensor_spec(value, TENSOR_CLASS_NAMES):
        return value
    if check_tensor_spec(value, "RaggedTensor"):
        return tf.RaggedTensorSpec.from_value(value)
    if check_tensor_spec(value, "SparseTensor"):
        return tf.SparseTensorSpec.from_value(value)
    if check_tensor_spec(value, "TensorArray"):
        return tf.TensorArraySpec.from_value(value)
    if check_tensor_spec(value, ("Tensor", "EagerTensor")):
        return tf.TensorSpec.from_tensor(value)
    raise BentoMLException(f"Unknown type for tensor spec, got{type(value)}.")


def get_input_signatures(
    func: "tf_ext.DecoratedFunction",
) -> t.Tuple["tf_ext.InputSignature"]:
    if hasattr(func, "function_spec"):  # RestoredFunction
        func_spec: "tf_ext.FunctionSpec" = getattr(func, "function_spec")
        input_spec: "tf_ext.TensorSignature" = getattr(func_spec, "input_signature")
        if input_spec is not None:
            return ((input_spec, {}),)
        else:
            concrete_func: t.List["tf_ext.ConcreteFunction"] = getattr(
                func, "concrete_functions"
            )
            return tuple(
                s for conc in concrete_func for s in get_input_signatures(conc)
            )
    else:
        sis: "tf_ext.InputSignature" = getattr(func, "structured_input_signature")
        if sis is not None:
            return (sis,)
        # NOTE: we can use internal `_arg_keywords` here.
        # Seems that this is a attributes of all ConcreteFunction and
        # does seem safe to access and use externally.
        if getattr(func, "_arg_keywords") is not None:
            return (
                (
                    tuple(),
                    {
                        k: normalize_spec(v)
                        for k, v in zip(
                            getattr(func, "_arg_keywords"), getattr(func, "inputs")
                        )
                    },
                ),
            )
    return tuple()


def get_output_signature(
    func: "tf_ext.DecoratedFunction",
) -> t.Union[
    "tf_ext.ConcreteFunction", t.Tuple[t.Any, ...], t.Dict[str, "tf_ext.TypeSpec"]
]:
    if hasattr(func, "function_spec"):  # for RestoredFunction
        # assume all concrete functions have same signature
        concrete_function_wrapper: "tf_ext.ConcreteFunction" = getattr(
            func, "concrete_functions"
        )[0]
        return get_output_signature(concrete_function_wrapper)

    if hasattr(func, "structured_input_signature"):  # for ConcreteFunction
        if getattr(func, "structured_outputs") is not None:
            outputs = getattr(func, "structured_outputs")
            if LazyType[t.Dict[str, "tf_ext.TensorSpec"]](dict).isinstance(outputs):
                return {k: normalize_spec(v) for k, v in outputs.items()}
            return outputs
        else:
            outputs: t.Tuple["tf_ext.TensorSpec"] = getattr(func, "outputs")
            return tuple(normalize_spec(v) for v in outputs)

    return tuple()


def get_arg_names(func: "tf_ext.DecoratedFunction") -> t.Optional[t.List[str]]:
    if hasattr(func, "function_spec"):  # for RestoredFunction
        func_spec: "tf_ext.FunctionSpec" = getattr(func, "function_spec")
        return getattr(func_spec, "arg_names")
    if hasattr(func, "structured_input_signature"):  # for ConcreteFunction
        return getattr(func, "_arg_keywords")
    return list()


def get_restored_functions(
    m: "tf_ext.Trackable",
) -> t.Dict[str, "tf_ext.RestoredFunction"]:
    function_map = {k: getattr(m, k) for k in dir(m)}
    return {
        k: v
        for k, v in function_map.items()
        if k not in TF_KERAS_DEFAULT_FUNCTIONS and hasattr(v, "function_spec")
    }


def get_serving_default_function(m: "tf_ext.Trackable") -> "tf_ext.ConcreteFunction":
    if not hasattr(m, "signatures"):
        raise EnvironmentError(f"{type(m)} is not a valid SavedModel format.")
    signatures: "tf_ext.SignatureMap" = getattr(m, "signatures")
    func = signatures.get(tf.compat.v2.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY)  # type: ignore
    if func is not None:
        return func
    raise BentoMLException(
        "Given Trackable objects doesn't contain a default functions from SignatureMap. Most likely Tensorflow internal error."
    )


def _pretty_format_function_call(base: str, name: str, arg_names: t.Tuple[t.Any]):
    if arg_names:
        part_sigs = ", ".join(f"{k}" for k in arg_names)
    else:
        part_sigs = ""

    if name == "__call__":
        return f"{base}({part_sigs})"
    return f"{base}.{name}({part_sigs})"


def _pretty_format_positional(positional: t.Optional["tf_ext.TensorSignature"]) -> str:
    if positional is not None:
        return f'Positional arguments ({len(positional)} total):\n    {"    * ".join(str(a) for a in positional)}'  # noqa
    return "No positional arguments.\n"


def pretty_format_function(
    function: "tf_ext.DecoratedFunction",
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
        positional, keyword = sig
        signature_descriptions.append(
            f"Arguments Option {index + 1}:\n  {_pretty_format_positional(positional)}\n  Keyword arguments:\n    {keyword}"
        )

    ret += "\n\n".join(signature_descriptions)
    ret += f"\n\nReturn:\n  {outs}\n\n"
    return ret


def pretty_format_restored_model(model: "tf_ext.AutoTrackable") -> str:
    part_functions = ""

    restored_functions = get_restored_functions(model)
    for name, func in restored_functions.items():
        part_functions += pretty_format_function(func, "model", name)
        part_functions += "\n"

    if get_tf_version().startswith("1"):
        serving_default = get_serving_default_function(model)
        if serving_default:
            part_functions += pretty_format_function(
                serving_default, "model", "signatures['serving_default']"
            )
            part_functions += "\n"

    return f"Found restored functions:\n{part_functions}"


def cast_tensor_by_spec(
    _input: "tf_ext.TensorLike", spec: "tf_ext.TypeSpec"
) -> "tf_ext.TensorLike":
    """
    transform dtype & shape following spec
    """
    if not LazyType["tf_ext.TensorSpec"](
        "tensorflow.python.framework.tensor_spec.TensorSpec"
    ).isinstance(spec):
        return _input

    if LazyType["tf_ext.CastableTensorType"]("tf.Tensor").isinstance(
        _input
    ) or LazyType["tf_ext.CastableTensorType"](
        "tensorflow.python.framework.ops.EagerTensor"
    ).isinstance(
        _input
    ):
        # TensorFlow Issues #43038
        # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        return t.cast(
            "tf_ext.TensorLike", tf.cast(_input, dtype=spec.dtype, name=spec.name)
        )
    else:
        return t.cast(
            "tf_ext.TensorLike", tf.constant(_input, dtype=spec.dtype, name=spec.name)
        )


class tf_function_wrapper:  # pragma: no cover
    def __init__(
        self,
        origin_func: t.Callable[..., t.Any],
        arg_names: t.Optional[t.List[str]] = None,
        arg_specs: t.Optional[t.Tuple["tf_ext.TensorSpec"]] = None,
        kwarg_specs: t.Optional[t.Dict[str, "tf_ext.TensorSpec"]] = None,
    ) -> None:
        self.origin_func = origin_func
        self.arg_names = arg_names
        self.arg_specs = arg_specs
        self.kwarg_specs = {k: v for k, v in zip(arg_names or [], arg_specs or [])}
        self.kwarg_specs.update(kwarg_specs or {})

    def __call__(
        self, *args: "tf_ext.TensorLike", **kwargs: "tf_ext.TensorLike"
    ) -> t.Any:
        if self.arg_specs is None and self.kwarg_specs is None:
            return self.origin_func(*args, **kwargs)

        for k in kwargs:
            if k not in self.kwarg_specs:
                raise TypeError(f"Function got an unexpected keyword argument {k}")

        arg_keys = {k for k, _ in zip(self.arg_names, args)}  # type: ignore[arg-type]
        _ambiguous_keys = arg_keys & set(kwargs)  # type: t.Set[str]
        if _ambiguous_keys:
            raise TypeError(f"got two values for arguments '{_ambiguous_keys}'")

        # INFO:
        # how signature with kwargs works?
        # https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/python/eager/function.py#L1519
        transformed_args: t.Tuple[t.Any, ...] = tuple(
            cast_tensor_by_spec(arg, spec) for arg, spec in zip(args, self.arg_specs)  # type: ignore[arg-type]
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
            arg_specs, kwarg_specs = sigs[0]
            setattr(
                loaded_model,
                k,
                cls(
                    func,
                    arg_names=arg_names,
                    arg_specs=arg_specs,  # type: ignore
                    kwarg_specs=kwarg_specs,  # type: ignore
                ),
            )
