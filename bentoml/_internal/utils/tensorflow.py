import typing as t
import importlib.util
from typing import TYPE_CHECKING

from .lazy_loader import LazyLoader

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

if TYPE_CHECKING:
    import tensorflow as tf
    from tensorflow.python.eager.function import ConcreteFunction
    from tensorflow.python.framework.type_spec import TypeSpec
    from tensorflow.python.training.tracking.base import Trackable
    from tensorflow.python.training.tracking.tracking import AutoTrackable
    from tensorflow.python.saved_model.signature_serialization import (
        _SignatureMap,  # type: ignore[reportPrivateUsage]
    )
    from tensorflow.python.saved_model.function_deserialization import RestoredFunction
else:
    tf = LazyLoader(
        "tf",
        globals(),
        "tensorflow",
        exc_msg=f"tensorflow is required to use {__name__}",
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


def _isinstance_wrapper(
    obj: t.Any, sobj: t.Union[str, type, t.Sequence[t.Any]]
) -> bool:
    """
    `isinstance` wrapper to check tensor spec

    Args:
        obj:
            tensor class to check.
        sobj:
            class used to check with :obj:`obj`. Follows `TENSOR_CLASS_NAME`

    Returns:
        :obj:`bool`
    """
    if not sobj:
        return False
    if isinstance(sobj, str):
        return type(obj).__name__ == sobj.split(".")[-1]
    if isinstance(sobj, (tuple, list, set)):
        return any(_isinstance_wrapper(obj, k) for k in sobj)
    return isinstance(obj, sobj)


def normalize_spec(value: t.Any) -> "TypeSpec":
    """normalize tensor spec"""
    if not _isinstance_wrapper(value, TENSOR_CLASS_NAMES):
        return value

    if _isinstance_wrapper(value, "RaggedTensor"):
        return tf.RaggedTensorSpec.from_value(value)
    if _isinstance_wrapper(value, "SparseTensor"):
        return tf.SparseTensorSpec.from_value(value)
    if _isinstance_wrapper(value, "TensorArray"):
        return tf.TensorArraySpec.from_tensor(value)
    if _isinstance_wrapper(value, ("Tensor", "EagerTensor")):
        return tf.TensorSpec.from_tensor(value)
    return value


def get_input_signatures(
    func: t.Union[t.Any, "RestoredFunction", "ConcreteFunction"]
) -> t.Tuple[t.Any, ...]:
    if hasattr(func, "function_spec"):  # for RestoredFunction
        if func.function_spec.input_signature:
            return ((func.function_spec.input_signature, {}),)
        else:
            return tuple(
                s
                for conc in func.concrete_functions
                for s in get_input_signatures(conc)
            )
    if hasattr(func, "structured_input_signature"):  # for ConcreteFunction
        if func.structured_input_signature is not None:
            return (func.structured_input_signature,)

        if func._arg_keywords is not None:  # TODO(bojiang): using private API
            return (
                (
                    tuple(),
                    {
                        k: normalize_spec(v)
                        for k, v in zip(func._arg_keywords, func.inputs)
                    },
                ),
            )
    return tuple()


def get_arg_names(
    func: t.Union[t.Any, "RestoredFunction", "ConcreteFunction"]
) -> t.Tuple[t.Any, ...]:
    if hasattr(func, "function_spec"):  # for RestoredFunction
        return func.function_spec.arg_names
    if hasattr(func, "structured_input_signature"):  # for ConcreteFunction
        return func._arg_keywords
    return tuple()


def get_output_signature(
    func: t.Union[t.Any, "RestoredFunction", "ConcreteFunction"]
) -> t.Tuple[t.Any, ...]:
    if hasattr(func, "function_spec"):  # for RestoredFunction
        # assume all concrete functions have same signature
        return get_output_signature(func.concrete_functions[0])

    if hasattr(func, "structured_input_signature"):  # for ConcreteFunction
        if func.structured_outputs is not None:
            if isinstance(func.structured_outputs, dict):
                return {
                    k: normalize_spec(v) for k, v in func.structured_outputs.items()
                }
            return func.structured_outputs
        else:
            return tuple(normalize_spec(v) for v in func.outputs)

    return tuple()


def get_restored_functions(m: "Trackable") -> t.Dict[str, t.Any]:
    function_map = {k: getattr(m, k, None) for k in dir(m)}
    return {
        k: v
        for k, v in function_map.items()
        if k not in TF_KERAS_DEFAULT_FUNCTIONS and hasattr(v, "function_spec")
    }


def get_serving_default_function(
    m: "Trackable",
) -> t.Callable[..., t.Any]:
    if not hasattr(m, "signatures"):
        raise EnvironmentError(f"{type(m)} is not a valid SavedModel format.")
    signatures: "_SignatureMap" = m.signatures
    return signatures.get(tf.compat.v2.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY)


def cast_tensor_by_spec(_input: "tf.Tensor", spec: "tf.TensorSpec") -> "tf.Tensor":
    """
    transform dtype & shape following spec
    """
    if not _isinstance_wrapper(spec, "TensorSpec"):
        return _input

    if _isinstance_wrapper(_input, ["Tensor", "EagerTensor"]):
        # TensorFlow issue #43038
        # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        return tf.cast(_input, dtype=spec.dtype, name=spec.name)
    else:
        return tf.constant(_input, dtype=spec.dtype, name=spec.name)


def _pretty_format_function_call(base: str, name: str, arg_names: t.Tuple[t.Any]):
    if arg_names:
        part_sigs = ", ".join(f"{k}" for k in arg_names)
    else:
        part_sigs = ""

    if name == "__call__":
        return f"{base}({part_sigs})"
    return f"{base}.{name}({part_sigs})"


def _pretty_format_positional(positional: t.List[t.Any]) -> str:
    return f'Positional arguments ({len(positional)} total):\n    * \n{"    * ".join(str(a) for a in positional)}'  # noqa


def pretty_format_function(
    function: t.Callable[..., t.Any], obj: str = "<object>", name: str = "<function>"
) -> str:
    ret = ""
    outs = get_output_signature(function)
    sigs = get_input_signatures(function)
    arg_names = get_arg_names(function)

    if hasattr(function, "function_spec"):
        arg_names = getattr(function, "function_spec").arg_names
    else:
        arg_names = function._arg_keywords  # type: ignore[reportFunctionMemberAccess]

    ret += _pretty_format_function_call(obj, name, arg_names)
    ret += "\n------------\n"

    signature_descriptions = []  # type: t.List[str]

    for index, sig in enumerate(sigs):
        positional, keyword = sig
        signature_descriptions.append(
            "Arguments Option {}:\n  {}\n  Keyword arguments:\n    {}".format(
                index + 1, _pretty_format_positional(positional), keyword
            )
        )

    ret += "\n\n".join(signature_descriptions)
    ret += f"\n\nReturn:\n  {outs}\n\n"
    return ret


def pretty_format_restored_model(model: "AutoTrackable") -> str:
    part_functions = ""

    restored_functions = get_restored_functions(model)
    for name, func in restored_functions.items():
        part_functions += pretty_format_function(func, "model", name)
        part_functions += "\n"

    serving_default = get_serving_default_function(model)
    if serving_default:
        part_functions += pretty_format_function(
            serving_default, "model", "signature['serving_default']"
        )
        part_functions += "\n"

    if not restored_functions and not serving_default:
        return (
            "No serving function was found in the saved model. "
            "In the model implementation, use `tf.function` decorator to mark "
            "the method needed for model serving. \n"
            "Find more details in related TensorFlow docs here "
            "https://www.tensorflow.org/api_docs/python/tf/saved_model/save"
        )
    return f"Found restored functions:\n{part_functions}"
