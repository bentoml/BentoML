from __future__ import annotations

import os
import typing as t
import tempfile

import onnx
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

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
    pytorch_model = make_pytorch_onnx_model(tmpdir)


onnx_pytorch_model = FrameworkTestModel(
    name="onnx_pytorch_model",
    model=pytorch_model,
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
models: list[FrameworkTestModel] = [onnx_pytorch_model]
