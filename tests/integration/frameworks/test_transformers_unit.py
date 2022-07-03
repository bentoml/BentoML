import typing as t
from typing import TYPE_CHECKING

import numpy as np
import pytest
import requests
from PIL import Image
from transformers import pipeline
from transformers.pipelines import SUPPORTED_TASKS
from transformers.trainer_utils import set_seed
from transformers.pipelines.base import Pipeline
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.modeling_tf_auto import TFAutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

import bentoml

if TYPE_CHECKING:
    from transformers.utils.generic import ModelOutput
    from transformers.tokenization_utils_base import BatchEncoding

    from bentoml._internal.external_typing import transformers as ext


set_seed(124)


@pytest.mark.parametrize(
    "name, pipeline, with_options, expected_options, input_data",
    [
        (
            "text-generation",
            pipeline(task="text-generation"),  # type: ignore
            {},
            {"task": "text-generation", "kwargs": {}},
            "A Bento box is a ",
        ),
        (
            "text-generation",
            pipeline(task="text-generation"),  # type: ignore
            {"kwargs": {"a": 1}},
            {"task": "text-generation", "kwargs": {"a": 1}},
            "A Bento box is a ",
        ),
        (
            "text-generation",
            tf_gpt2_pipeline(),
            {},
            {"task": "text-generation", "kwargs": {}},
            "A Bento box is a ",
        ),
        (
            "text-generation",
            pt_gpt2_pipeline(),
            {},
            {"task": "text-generation", "kwargs": {}},
            "A Bento box is a ",
        ),
        (
            "image-classification",
            pipeline("image-classification"),  # type: ignore
            {},
            {"task": "image-classification", "kwargs": {}},
            Image.open(
                requests.get(
                    "http://images.cocodataset.org/val2017/000000039769.jpg",
                    stream=True,
                ).raw
            ),
        ),
        (
            "text-classification",
            pipeline("text-classification"),  # type: ignore
            {},
            {"task": "text-classification", "kwargs": {}},
            "BentoML is an awesome library for machine learning.",
        ),
    ],
)
def test_transformers(
    name: str,
    pipeline: "ext.TransformersPipelineType",  # type: ignore
    with_options: t.Dict[str, t.Any],
    expected_options: t.Dict[str, t.Any],
    input_data: t.Any,
):
    saved_model = bentoml.transformers.save_model(name, pipeline)
    assert saved_model is not None
    assert saved_model.tag.name == name

    bento_model: bentoml.Model = saved_model.with_options(**with_options)
    assert bento_model.tag == saved_model.tag
    assert bento_model.info.context.framework_name == "transformers"
    assert bento_model.info.options.task == expected_options["task"]  # type: ignore
    assert bento_model.info.options.kwargs == expected_options["kwargs"]  # type: ignore

    runnable: bentoml.Runnable = bentoml.transformers.get_runnable(bento_model)()
    output_data = runnable(input_data)  # type: ignore
    assert output_data is not None


class CustomPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs: t.Any) -> tuple[AnyDict, AnyDict, AnyDict]:
        preprocess_kwargs: AnyDict = {}
        if "dummy_arg" in kwargs:
            preprocess_kwargs["dummy_arg"] = kwargs["dummy_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text: str, dummy_arg: int = 2) -> BatchEncoding | None:
        if self.tokenizer:
            input_ids = self.tokenizer(text, return_tensors="pt")
            return input_ids

    def postprocess(self, model_outputs: ModelOutput, **parameters: AnyDict) -> t.Any:
        return outputs["logits"].softmax(-1).numpy()  # type: ignore (unfinished transformers type)


def test_custom_pipeline():
    TASK_NAME: str = "my-classification-task"
    TASK_DEFINITION: t.Dict[str, t.Any] = {
        "impl": CustomPipeline,
        "tf": (),
        "pt": (AutoModelForSequenceClassification,),
        "default": {},
        "type": "text",
    }
    SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION

    pipe = pipeline(
        task=TASK_NAME,
        model=AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        ),
        tokenizer=AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        ),
    )

    saved_pipe = bentoml.transformers.save_model(
        "my_classification_model",
        pipeline=pipe,
        task_name=TASK_NAME,
        task_definition=TASK_DEFINITION,
    )

    assert saved_pipe is not None
    assert saved_pipe.tag.name == "my_classification_model"

    # Remove the task definition from the list of Transformers supported tasks
    del SUPPORTED_TASKS[TASK_NAME]

    input_data: t.List[str] = [
        "BentoML: Create an ML Powered Prediction Service in Minutes via @TDataScience https://buff.ly/3srhTw9 #Python #MachineLearning #BentoML",
        "Top MLOps Serving frameworks — 2021 https://link.medium.com/5Elq6Aw52ib #mlops #TritonInferenceServer #opensource #nvidia #machincelearning  #serving #tensorflow #PyTorch #Bodywork #BentoML #KFServing #kubeflow #Cortex #Seldon #Sagify #Syndicai",
        "#MLFlow provides components for experimentation management, ML project management. #BentoML only focuses on serving and deploying trained models",
        "2000 and beyond #OpenSource #bentoml",
        "Model Serving Made Easy https://github.com/bentoml/BentoML ⭐ 1.1K #Python #Bentoml #BentoML #Modelserving #Modeldeployment #Modelmanagement #Mlplatform #Mlinfrastructure #Ml #Ai #Machinelearning #Awssagemaker #Awslambda #Azureml #Mlops #Aiops #Machinelearningoperations #Turn",
    ]
    pipe = bentoml.transformers.load_model("my_classification_model:latest")
    output_data = pipe(input_data)
    assert output_data is not None
    assert isinstance(output_data, list) and len(output_data) == len(input_data)
    assert all([isinstance(data, np.ndarray) for data in output_data])

    runnable: bentoml.Runnable = bentoml.transformers.get_runnable(saved_pipe)()
    output_data = runnable(input_data)
    assert output_data is not None
    assert isinstance(output_data, list) and len(output_data) == len(input_data)
    assert all([isinstance(data, np.ndarray) for data in output_data])
