from __future__ import annotations

import typing as t
import logging

import pytest
import transformers
from transformers.pipelines import pipeline  # type: ignore
from transformers.pipelines import check_task  # type: ignore
from transformers.trainer_utils import set_seed
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.modeling_auto import AutoModelForAudioClassification
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.pipelines.audio_classification import AudioClassificationPipeline

import bentoml
from bentoml.exceptions import BentoMLException
from bentoml.transformers import ModelOptions

if t.TYPE_CHECKING:
    from bentoml._internal.frameworks.transformers import TaskDefinition


set_seed(124)


def test_convert_to_auto_class():
    from bentoml._internal.frameworks.transformers import convert_to_autoclass

    with pytest.raises(
        BentoMLException, match="Given not_a_class is not a valid Transformers *"
    ):
        convert_to_autoclass("not_a_class")

    assert (
        convert_to_autoclass("AutoModelForSequenceClassification")
        is AutoModelForSequenceClassification
    )
    assert convert_to_autoclass("AutoModelForCausalLM") is AutoModelForCausalLM


@pytest.fixture(name="sentiment_task")
def fixture_sentiment() -> tuple[transformers.Pipeline, TaskDefinition]:
    alias, original_task = t.cast(
        "tuple[t.Literal['text-classification'], TaskDefinition, t.Any]",
        check_task("sentiment-analysis"),
    )[:2]
    return (
        pipeline(alias, model="hf-internal-testing/tiny-random-distilbert"),
        original_task,
    )


def test_raise_different_default_definition(
    sentiment_task: tuple[transformers.Pipeline, TaskDefinition]
):
    # implementation is different
    sentiment, _ = sentiment_task
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

    with pytest.raises(bentoml.exceptions.BentoMLException) as exc_info:
        _ = bentoml.transformers.save_model(
            "forbidden_override",
            sentiment,
            task_name="sentiment-analysis",
            task_definition=task_definition,
        )
        assert "does not match pipeline task definition" in str(exc_info.value)


def test_raise_does_not_match_task_name(
    sentiment_task: tuple[transformers.Pipeline, TaskDefinition]
):
    # pipeline task does not match given task name or pipeline.task is None
    sentiment, original_task = sentiment_task

    with pytest.raises(
        bentoml.exceptions.BentoMLException,
        match=f"Argument 'task_name' 'custom' does not match pipeline task name '{sentiment.task}'.",
    ):
        _ = bentoml.transformers.save_model(
            "forbidden_override",
            sentiment,
            task_name="custom",
            task_definition=original_task,
        )


def test_raise_does_not_match_impl_field(
    sentiment_task: tuple[transformers.Pipeline, TaskDefinition]
):
    sentiment, original_task = sentiment_task
    # task_definition['impl'] is different from pipeline type
    with pytest.raises(
        bentoml.exceptions.BentoMLException,
        match=f"Argument 'pipeline' is not an instance of {AudioClassificationPipeline}. It is an instance of {type(sentiment)}.",
    ):
        original_task["impl"] = AudioClassificationPipeline
        _ = bentoml.transformers.save_model(
            "forbidden_override",
            sentiment,
            task_name="text-classification",
            task_definition=original_task,
        )


def test_raises_is_not_pipeline_instance():
    with pytest.raises(bentoml.exceptions.BentoMLException) as exc_info:
        _ = bentoml.transformers.save_model(
            "not_pipeline_type", AudioClassificationPipeline
        )
    assert (
        "'pipeline' must be an instance of 'transformers.pipelines.base.Pipeline'. "
        in str(exc_info.value)
    )


def test_logs_custom_task_definition(
    caplog: pytest.LogCaptureFixture,
    sentiment_task: tuple[transformers.Pipeline, TaskDefinition],
):
    sentiment, original_task = sentiment_task
    with caplog.at_level(logging.INFO):
        _ = bentoml.transformers.save_model(
            "custom_sentiment_pipeline",
            sentiment,
            task_name=sentiment.task,
            task_definition=original_task,
        )
    assert (
        "Arguments 'task_name' and 'task_definition' are provided. Saving model with pipeline "
        in caplog.text
    )


def test_log_load_model(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.DEBUG):
        bento_model = bentoml.transformers.save_model(
            "sentiment_test",
            pipeline(
                task="text-classification",
                model="hf-internal-testing/tiny-random-distilbert",
            ),
        )
        _ = bentoml.transformers.load_model("sentiment_test:latest", use_fast=True)
    assert (
        f"Loading '{t.cast(ModelOptions, bento_model.info.options).task!s}' pipeline (tag='{bento_model.tag!s}')"
        in caplog.text
    )


def test_model_options():
    unstructured_options: t.Dict[str, t.Any] = {
        "task": "sentiment-analysis",
        "tf": (),
        "pt": (),
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
