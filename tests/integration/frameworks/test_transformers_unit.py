from __future__ import annotations

import logging

import pytest
from transformers.pipelines import pipeline  # type: ignore
from transformers.pipelines import check_task  # type: ignore
from transformers.trainer_utils import set_seed
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.models.auto.modeling_auto import AutoModelForAudioClassification
from transformers.pipelines.audio_classification import AudioClassificationPipeline

import bentoml
from bentoml.exceptions import BentoMLException

set_seed(124)


def test_get_auto_class():
    from bentoml._internal.frameworks.transformers import _get_auto_class

    with pytest.raises(BentoMLException) as exc_info:
        _ = [i for i in _get_auto_class("not_a_class")]
    assert "neither exists nor a valid Transformers auto class." in str(exc_info.value)

    with pytest.raises(BentoMLException) as exc_info:
        _ = [i for i in _get_auto_class(["not_a_class"])]
    assert "neither exists nor a valid Transformers auto class." in str(exc_info.value)

    with pytest.raises(
        BentoMLException,
        match=f"Unsupported type {type(1)}. Only support str | Iterable[str].",
    ) as exc_info:
        _ = [i for i in _get_auto_class(1)]  # type: ignore (testing invalid type)

    for klass in _get_auto_class(
        ["AutoModelForSequenceClassification", "AutoModelForCausalLM"]
    ):
        assert issubclass(klass, _BaseAutoModelClass)


alias = "sentiment-analysis"
original_task, _ = check_task(alias)  # type: ignore (unfinished transformers type)
sentiment = pipeline(alias, model="hf-internal-testing/tiny-random-distilbert")


def test_raise_different_default_definition():

    # implementation is different
    task_definition = (
        {
            "impl": AudioClassificationPipeline,
            "tf": (),
            "pt": (AutoModelForAudioClassification,),
            "default": {
                "model": {
                    "pt": ("hf-internal-testing/tiny-random-distilbert",),
                },
            },
            "type": "text",
        },
    )

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
        match=f"Argument `task_name` 'custom' does not match pipeline task name '{sentiment.task}'.",
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
            match=f"Argument `pipeline` is not an instance of {AudioClassificationPipeline}. It is an instance of {type(sentiment)}.",
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
        "`pipeline` must be an instance of `transformers.pipelines.base.Pipeline`. "
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
        "Arguments `task_name` and `task_definition` are provided. Saving model with pipeline "
        in caplog.text
    )
