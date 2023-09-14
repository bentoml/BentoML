from __future__ import annotations

import bentoml


def test_import_model_with_synced_version():
    revision = "1b6409d3a7b346630b8cf0eeadc692e73779be49"

    bento_model = bentoml.diffusers.import_model(
        "tiny-sd",
        "hf-internal-testing/tiny-stable-diffusion-torch",
        sync_with_hub_version=True,
        revision=revision,
    )

    assert bento_model.tag.version == revision

    revision = "895564fe990c3e443580679ac0ad2958f09b2c67"

    bento_model = bentoml.diffusers.import_model(
        "tiny-sd:asdf",
        "hf-internal-testing/tiny-stable-diffusion-torch",
        sync_with_hub_version=True,
        revision=revision,
    )

    assert bento_model.tag.version == revision

    revision = "895564fe990c3e443580679ac0ad2958f09b2c67"

    bento_model = bentoml.diffusers.import_model(
        "tiny-sd:asdf",
        "hf-internal-testing/tiny-stable-diffusion-torch",
        revision=revision,
    )

    assert bento_model.tag.version == "asdf"
