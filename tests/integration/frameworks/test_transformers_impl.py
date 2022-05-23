import typing as t
from typing import TYPE_CHECKING

import pytest
import requests
import transformers
import transformers.pipelines
from transformers.trainer_utils import set_seed
from PIL import Image

import bentoml

if TYPE_CHECKING:
    from bentoml._internal.external_typing import transformers as ext


set_seed(124)


@pytest.mark.parametrize(
    "name, pipeline, model_options, input_data",
    [
        (
            "text-generation",
            transformers.pipeline(task="text-generation"), # type: ignore
            {"pipeline": True, "task": "text-generation"},
            "A Bento box is a ",
        ),
        (
            "image-classification",
            transformers.pipeline("image-classification"), # type: ignore
            {"pipeline": True, "task": "image-classification"},
            Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
        ),
        (
            "text-classification",
            transformers.pipeline("text-classification"), # type: ignore
            {"pipeline": True, "task": "text-classification"},
            "BentoML is an awesome library for machine learning.",
        ),
    ],
)
def test_transformers(
    name: str,
    pipeline: "ext.TransformersPipelineType", # type: ignore
    model_options: t.Dict[str, t.Any],
    input_data: t.Any,
):
    tag: bentoml.Tag = bentoml.transformers.save_model(name, pipeline)
    assert tag is not None
    assert tag.name == name

    bento_model: bentoml.Model = bentoml.transformers.get(tag)
    assert bento_model.tag == tag
    assert bento_model.info.context.framework_name == "transformers"
    assert dict(bento_model.info.options) == model_options

    runnable: bentoml.Runnable = bentoml.transformers.get_runnable(bento_model)()
    output_data = runnable(input_data) # type: ignore
    assert output_data is not None
