from __future__ import annotations

import pytest
import transformers

import bentoml


@pytest.mark.parametrize(
    "extractor_cls",
    [transformers.WhisperFeatureExtractor, transformers.Wav2Vec2FeatureExtractor],
)
def test_save_audio_feature_extractor(extractor_cls):
    extractor = extractor_cls(sampling_rate=22050)

    model = bentoml.transformers.save_model("audio_feature_extractor", extractor)

    assert set(model.info.signatures) == {"pad"}
    restored = bentoml.transformers.load_model(model)
    assert isinstance(restored, extractor_cls)
    assert restored.sampling_rate == extractor.sampling_rate


def test_save_image_processor():
    pytest.importorskip("PIL")
    processor = transformers.ViTImageProcessor()

    model = bentoml.transformers.save_model("image_processor", processor)

    assert set(model.info.signatures) == {"__call__", "preprocess"}
    restored = bentoml.transformers.load_model(model)
    assert isinstance(restored, transformers.ViTImageProcessor)
    assert restored.to_dict() == processor.to_dict()
