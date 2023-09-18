from __future__ import annotations

import numpy as np
import diffusers

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.diffusers

backward_compatible = False


def check_output(out):
    # output is a tuple of (images, _)
    arr = out[0][0]
    return arr.shape == (256, 256, 3)


pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
    "hf-internal-testing/tiny-stable-diffusion-torch"
)

diffusers_model = FrameworkTestModel(
    name="diffusers",
    model=pipeline,
    configurations=[
        Config(
            test_inputs={
                "__call__": [
                    Input(
                        input_args=[],
                        input_kwargs={
                            "prompt": "a bento box",
                            "width": 256,
                            "height": 256,
                            "num_inference_steps": 3,
                            "output_type": np,
                        },
                        expected=check_output,
                    )
                ],
            },
        ),
    ],
)


models: list[FrameworkTestModel] = [diffusers_model]
