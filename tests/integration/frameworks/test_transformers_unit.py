from __future__ import annotations

import typing as t
import logging

import pytest
from transformers.pipelines import pipeline  # type: ignore
from transformers.pipelines import check_task  # type: ignore
from transformers.trainer_utils import set_seed
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.modeling_auto import AutoModelForAudioClassification
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines.audio_classification import AudioClassificationPipeline

import bentoml
from bentoml.exceptions import BentoMLException
from bentoml.transformers import ModelOptions

set_seed(124)


def test_convert_to_auto_class():
    from bentoml._internal.frameworks.transformers import (
        _convert_to_auto_class,  # type: ignore
    )

    with pytest.raises(
        BentoMLException, match="Given not_a_class is not a valid Transformers *"
    ):
        _convert_to_auto_class("not_a_class")

    assert (
        _convert_to_auto_class("AutoModelForSequenceClassification")
        is AutoModelForSequenceClassification
    )
    assert _convert_to_auto_class("AutoModelForCausalLM") is AutoModelForCausalLM


alias = "sentiment-analysis"
_, original_task, _ = check_task(alias)  # type: ignore (unfinished transformers type)
sentiment = pipeline(alias, model="hf-internal-testing/tiny-random-distilbert")


def test_raise_different_default_definition():

    # implementation is different
    task_definition = {
        "impl": AudioClassificationPipeline,
        "tf": (),
        "pt": (AutoModelForAudioClassification,),
        "default": {
            "model": {
                "pt": ("hf-internal-testing/tiny-random-distilbert",),
            },
        },
        "type": "text",
    }

    with pytest.raises(BentoMLException) as exc_info:
        _ = bentoml.transformers.save_model(
            "forbidden_override",
            sentiment,
            task_name=alias,
            task_definition=task_definition,  # type: ignore (testing invalid type)
        )
        assert "does not match pipeline task definition" in str(exc_info.value)


def test_raise_does_not_match_task_name():
    # pipeline task does not match given task name or pipeline.task is None
    with pytest.raises(
        BentoMLException,
        match=f"Argument 'task_name' 'custom' does not match pipeline task name '{sentiment.task}'.",
    ):
        _ = bentoml.transformers.save_model(
            "forbidden_override",
            sentiment,
            task_name="custom",
            task_definition=original_task,  # type: ignore (unfinished transformers type)
        )


def test_raise_does_not_match_impl_field():
    # task_definition['impl'] is different from pipeline type
    orig_impl: type = original_task["impl"]
    try:
        with pytest.raises(
            BentoMLException,
            match=f"Argument 'pipeline' is not an instance of {AudioClassificationPipeline}. It is an instance of {type(sentiment)}.",
        ):
            original_task["impl"] = AudioClassificationPipeline
            _ = bentoml.transformers.save_model(
                "forbidden_override",
                sentiment,
                task_name=alias,
                task_definition=original_task,  # type: ignore (unfinished transformers type)
            )
    finally:
        original_task["impl"] = orig_impl  # type: ignore (unfinished transformers type)


def test_raises_is_not_pipeline_instance():
    with pytest.raises(BentoMLException) as exc_info:
        _ = bentoml.transformers.save_model(
            "not_pipeline_type", AudioClassificationPipeline  # type: ignore (testing invalid type)
        )
    assert (
        "'pipeline' must be an instance of 'transformers.pipelines.base.Pipeline'. "
        in str(exc_info.value)
    )


def test_logs_custom_task_definition(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        _ = bentoml.transformers.save_model(
            "custom_sentiment_pipeline",
            sentiment,
            task_name="sentiment-analysis",
            task_definition=original_task,  # type: ignore (unfinished transformers type)
        )
    assert (
        "Arguments 'task_name' and 'task_definition' are provided. Saving model with pipeline "
        in caplog.text
    )


def test_log_load_model(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        _ = bentoml.transformers.save_model(
            "sentiment_test",
            pipeline(
                task="text-classification",
                model="hf-internal-testing/tiny-random-distilbert",
            ),
        )
        _ = bentoml.transformers.load_model("sentiment_test:latest", use_fast=True)
    assert "with kwargs {'use_fast': True}." in caplog.text


def test_model_options():
    unstructured_options: t.Dict[str, t.Any] = {
        "task": "sentiment-analysis",
        "tf": [],
        "pt": [],
        "default": {},
        "type": None,
        "kwargs": {},
    }

    structured_options = ModelOptions(task="sentiment-analysis")
    assert structured_options.to_dict() == unstructured_options

    structured_options = ModelOptions(**unstructured_options)
    assert structured_options is not None
    assert structured_options.tf == ()
    assert structured_options.pt == ()
    assert structured_options.default == {}
    assert structured_options.type is None
    assert structured_options.kwargs == {}


def test_custom_pipeline():
    """
    Test saving and loading a custom pipeline from scratch. This is not covered by the framework tests
    because it does not load a custom pipeline without first adding to the transformers SUPPORTED_TASKS.
    """
    import numpy as np
    from transformers.pipelines import SUPPORTED_TASKS

    from .models.transformers import CustomPipeline
    from .models.transformers import TINY_TEXT_MODEL

    TASK_NAME: str = "custom-classification-task"
    TASK_DEFINITION: t.Dict[str, t.Any] = {
        "impl": CustomPipeline,
        "tf": (),
        "pt": (AutoModelForSequenceClassification,),
        "default": {},
        "type": "text",
    }

    # Ensure that the task is not already registered

    try:
        SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION

        pipe = pipeline(
            task=TASK_NAME,
            model=AutoModelForSequenceClassification.from_pretrained(TINY_TEXT_MODEL),
            tokenizer=AutoTokenizer.from_pretrained(TINY_TEXT_MODEL),
        )

        saved_pipe = bentoml.transformers.save_model(
            "my_classification_model",
            pipeline=pipe,
            task_name=TASK_NAME,
            task_definition=TASK_DEFINITION,
        )

        assert saved_pipe is not None
        assert saved_pipe.tag.name == "my_classification_model"
    finally:
        # Remove the task definition from the list of Transformers supported tasks
        del SUPPORTED_TASKS[TASK_NAME]
    assert TASK_NAME not in SUPPORTED_TASKS

    try:
        SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION

        pipe = pipeline(
            task=TASK_NAME,
            model=AutoModelForSequenceClassification.from_pretrained(TINY_TEXT_MODEL),
            tokenizer=AutoTokenizer.from_pretrained(TINY_TEXT_MODEL),
        )

        saved_pipe = bentoml.transformers.save_model(
            "my_classification_model",
            pipeline=pipe,
            task_name=TASK_NAME,
            task_definition=TASK_DEFINITION,
        )
    finally:
        if TASK_NAME in SUPPORTED_TASKS:
            del SUPPORTED_TASKS[TASK_NAME]
        assert TASK_NAME not in SUPPORTED_TASKS

    assert saved_pipe is not None
    assert saved_pipe.tag.name == "my_classification_model"

    input_data: t.List[str] = [
        "BentoML: Create an ML Powered Prediction Service in Minutes via @TDataScience https://buff.ly/3srhTw9 #Python #MachineLearning #BentoML",
        "Top MLOps Serving frameworks — 2021 https://link.medium.com/5Elq6Aw52ib #mlops #TritonInferenceServer #opensource #nvidia #machincelearning  #serving #tensorflow #PyTorch #Bodywork #BentoML #KFServing #kubeflow #Cortex #Seldon #Sagify #Syndicai",
        "#MLFlow provides components for experimentation management, ML project management. #BentoML only focuses on serving and deploying trained models",
        "2000 and beyond #OpenSource #bentoml",
        "Model Serving Made Easy https://github.com/bentoml/BentoML ⭐ 1.1K #Python #Bentoml #BentoML #Modelserving #Modeldeployment #Modelmanagement #Mlplatform #Mlinfrastructure #Ml #Ai #Machinelearning #Awssagemaker #Awslambda #Azureml #Mlops #Aiops #Machinelearningoperations #Turn",
    ]

    try:
        pipe = bentoml.transformers.load_model("my_classification_model:latest")
        output_data = pipe(input_data)
    finally:
        del SUPPORTED_TASKS[TASK_NAME]
        assert TASK_NAME not in SUPPORTED_TASKS

    assert output_data is not None
    assert isinstance(output_data, list) and len(output_data) == len(input_data)
    assert all([isinstance(data, np.ndarray) for data in output_data])

    try:
        runnable: bentoml.Runnable = bentoml.transformers.get_runnable(saved_pipe)()
        output_data = runnable(input_data)
    finally:
        del SUPPORTED_TASKS[TASK_NAME]
        assert TASK_NAME not in SUPPORTED_TASKS

    assert output_data is not None
    assert isinstance(output_data, list) and len(output_data) == len(input_data)
    assert all([isinstance(data, np.ndarray) for data in output_data])
