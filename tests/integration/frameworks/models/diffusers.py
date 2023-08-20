from __future__ import annotations

import diffusers
import numpy as np

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelConfiguration as Config
from . import FrameworkTestModelInput as Input

framework = bentoml.diffusers

backward_compatible = False


def check_output(out):
    # output is a tuple of (images, _)
    arr = out[0][0]
    return arr.shape == (256, 256, 3)


def check_replace_scheduler_factory(expected_output):
    def _check(d):
        return d == expected_output

    return _check


replace_success = {"success": True}
replace_import_failure = {
    "success": False,
    "error_message": "cannot import scheduler class",
}
replace_incompatible_failure = {
    "success": False,
    "error_message": "scheduler class is incompatible to this pipeline",
}


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
        Config(
            test_inputs={
                "_replace_scheduler": [
                    Input(
                        input_args=[
                            "diffusers.schedulers.scheduling_dpmsolver_multistep.NonExistScheduler"
                        ],
                        expected=check_replace_scheduler_factory(
                            replace_import_failure
                        ),
                    ),
                    Input(
                        input_args=[
                            "diffusers.schedulers.nonexist_module.NonExistScheduler"
                        ],
                        expected=check_replace_scheduler_factory(
                            replace_import_failure
                        ),
                    ),
                    Input(
                        input_args=[
                            "diffusers.schedulers.scheduling_repaint.RePaintSchedulerOutput"
                        ],
                        expected=check_replace_scheduler_factory(
                            replace_incompatible_failure
                        ),
                    ),
                    Input(
                        input_args=[
                            "diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler"
                        ],
                        expected=check_replace_scheduler_factory(replace_success),
                    ),
                ],
            },
        ),
    ],
)


models: list[FrameworkTestModel] = [diffusers_model]
