from __future__ import annotations

import os
import typing as t
import tempfile
from typing import TYPE_CHECKING

import onnx
import numpy as np
import torch
import sklearn
import torch.nn as nn
import onnxruntime as ort
from skl2onnx import convert_sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.data_types import Int64TensorType
from skl2onnx.common.data_types import StringTensorType

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

if TYPE_CHECKING:
    import bentoml._internal.external_typing as ext

framework = bentoml.onnx

backward_compatible = True

# specify parameters via map
param = {"max_depth": 3, "eta": 0.3, "objective": "multi:softprob", "num_class": 2}


def method_caller(
    framework_test_model: FrameworkTestModel,
    method: str,
    args: list[t.Any],
    kwargs: dict[str, t.Any],
):
    with tempfile.NamedTemporaryFile() as temp:
        onnx.save(framework_test_model.model, temp.name)
        ort_sess = ort.InferenceSession(temp.name, providers=["CPUExecutionProvider"])

    def to_numpy(item):
        if isinstance(item, np.ndarray):
            pass
        elif isinstance(item, torch.Tensor):
            item = item.detach().to("cpu").numpy()
        return item

    input_names = {i.name: to_numpy(val) for i, val in zip(ort_sess.get_inputs(), args)}
    output_names = [o.name for o in ort_sess.get_outputs()]
    out = getattr(ort_sess, method)(output_names, input_names)[0]
    return out


def check_model(model: bentoml.Model, resource_cfg: dict[str, t.Any]):
    from bentoml._internal.resource import get_resource

    if get_resource(resource_cfg, "nvidia.com/gpu"):
        pass
    elif get_resource(resource_cfg, "cpu"):
        cpus = round(get_resource(resource_cfg, "cpu"))
        assert model._providers == ["CPUExecutionProvider"]
        assert model._sess_options.inter_op_num_threads == cpus
        assert model._sess_options.intra_op_num_threads == cpus


def close_to(expected):
    def check_output(out):
        return np.isclose(out, expected, rtol=1e-03).all()

    return check_output


N, D_in, H, D_out = 64, 1000, 100, 1


class PyTorchModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(PyTorchModel, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


pytorch_input = torch.randn(N, D_in)
pytorch_model = PyTorchModel(D_in, H, D_out)
pytorch_expected = pytorch_model(pytorch_input).detach().to("cpu").numpy()


def make_pytorch_onnx_model(tmpdir):
    input_names = ["x"]
    output_names = ["output1"]
    model_path = os.path.join(tmpdir, "pytorch.onnx")
    torch.onnx.export(
        pytorch_model,
        pytorch_input,
        model_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "x": {0: "batch_size"},  # variable length axes
            "output1": {0: "batch_size"},
        },
    )
    onnx_model = onnx.load(model_path)
    return onnx_model


with tempfile.TemporaryDirectory() as tmpdir:
    onnx_pytorch_raw_model = make_pytorch_onnx_model(tmpdir)


onnx_pytorch_model = FrameworkTestModel(
    name="onnx_pytorch_model",
    model=onnx_pytorch_raw_model,
    model_method_caller=method_caller,
    model_signatures={"run": {"batchable": True}},
    configurations=[
        Config(
            test_inputs={
                "run": [
                    Input(
                        input_args=[pytorch_input],
                        expected=close_to(pytorch_expected),
                    ),
                ],
            },
            check_model=check_model,
        ),
    ],
)


# sklearn random forest with multiple outputs
def make_rf_onnx_model() -> (
    tuple[onnx.ModelProto, tuple[ext.NpNDArray, tuple[ext.NpNDArray, ext.NpNDArray]]]
):
    iris: sklearn.utils.Bunch = load_iris()
    X: ext.NpNDArray = iris.data
    y: ext.NpNDArray = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onnx_model = t.cast(
        onnx.ModelProto, convert_sklearn(clr, initial_types=initial_type)
    )
    expected_input = t.cast("ext.NpNDArray", X_test[:2])
    expected_output1 = t.cast("ext.NpNDArray", clr.predict(expected_input))
    expected_output2 = t.cast("ext.NpNDArray", clr.predict_proba(expected_input))
    expected_output = (expected_output1, expected_output2)
    expected_data = (expected_input, expected_output)
    return (onnx_model, expected_data)


# the output of onnxruntime has a different format from the output of
# the original model, we need generate a function to adapt the outputs
# of onnxruntime (also the BentoML runner) to the outputs of original
# model
def gen_rf_output_checker(
    expected_output: tuple[ext.NpNDArray, ext.NpNDArray]
) -> t.Callable[[t.Any], bool]:
    expected_output1, expected_output2 = expected_output

    def _check(out: tuple[ext.NpNDArray, list[dict[int, float]]]) -> bool:
        out1, out2 = out
        flag1 = (out1 == expected_output1).all()
        out2_lst = [[d[idx] for idx in sorted(d.keys())] for d in out2]
        flag2 = t.cast(
            bool, np.isclose(np.array(out2_lst), expected_output2, rtol=1e-3).all()
        )
        return flag1 and flag2

    return _check


onnx_rf_raw_model, _expected_data = make_rf_onnx_model()
rf_input, rf_expected_output = _expected_data

onnx_rf_model = FrameworkTestModel(
    name="onnx_rf_model",
    model=onnx_rf_raw_model,
    model_method_caller=method_caller,
    model_signatures={"run": {"batchable": True}},
    configurations=[
        Config(
            test_inputs={
                "run": [
                    Input(
                        input_args=[rf_input],
                        expected=gen_rf_output_checker(rf_expected_output),
                    ),
                ],
            },
            check_model=check_model,
        ),
    ],
)


# sklearn label encoder testing int and string input types
LT = t.TypeVar("LT")


def make_le_onnx_model(
    labels: list[LT], tensor_type: type
) -> tuple[onnx.ModelProto, tuple[list[list[LT]], ext.NpNDArray]]:
    le = LabelEncoder()
    le.fit(labels)

    initial_type = [("tensor_input", tensor_type([None, 1]))]
    onnx_model = t.cast(
        onnx.ModelProto, convert_sklearn(le, initial_types=initial_type)
    )
    expected_input = [[labels[0]], [labels[1]]]

    expected_output = t.cast("ext.NpNDArray", le.transform(expected_input))
    expected_data = (expected_input, expected_output)
    return (onnx_model, expected_data)


onnx_le_models = []
int_labels = [5, 2, 3]
str_labels = ["apple", "orange", "cat"]

for labels, tensor_type in [
    (int_labels, Int64TensorType),
    (str_labels, StringTensorType),
]:
    onnx_le_raw_model, expected_data = make_le_onnx_model(labels, tensor_type)
    le_input, le_expected_output = expected_data

    def _check(
        out: ext.NpNDArray, expected_out: ext.NpNDArray = le_expected_output
    ) -> bool:
        # LabelEncoder's raw output have one less dim than the onnxruntime's output
        flat_out = np.squeeze(out, axis=1)
        return (expected_out == flat_out).all()

    onnx_le_model = FrameworkTestModel(
        name=f"onnx_le_model_{tensor_type.__name__.lower()}",
        model=onnx_le_raw_model,
        model_method_caller=method_caller,
        model_signatures={"run": {"batchable": True}},
        configurations=[
            Config(
                test_inputs={
                    "run": [
                        Input(
                            input_args=[le_input],
                            expected=_check,
                        ),
                    ],
                },
                check_model=check_model,
            ),
        ],
    )
    onnx_le_models.append(onnx_le_model)


models: list[FrameworkTestModel] = [onnx_pytorch_model, onnx_rf_model] + onnx_le_models
