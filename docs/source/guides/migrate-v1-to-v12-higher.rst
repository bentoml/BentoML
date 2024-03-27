===========
Migrate v1.1.x to v1.2.x higher
===========

You can see the more detail about BentoML 1.2 in Blog Post `Introducing BentoML 1.2 <https://www.bentoml.com/blog/introducing-bentoml-1-2/>`_.




QuickStart
------------------------------------

you can still use ``BentoML v1.x.x Style`` even if with ``BentoML v1.2.x higher`` now.

.. tab-set::

    .. tab-item:: BentoML v1.x.x Style

        In this case(``BentoML v1.x.x``), you can make ModelServer with BentoML two Core Component ``Service`` and ``Runner``.
        one BentoML Service can have Runners 0 or more than one

        .. code-block:: python

            # bentoml v1.x.x Style
            # service.py
            import bentoml
            import numpy as np
            import torch
            from numpy import typing as tnp
            from pydantic import Field

            sample_model_runner = bentoml.models.get("hello-bento-samplemodel:2024-03-17").to_runner()

            svc = bentoml.Service(name="hello-bento-service", runners=[sample_model_runner])

            INPUT_EXAMPLE = [
                [1.0, 0.3345, 0.4554, 0.4435],
                [1.0, 0.3345, 0.4554, 0.4435],
            ]
            OUTPUT_EXAMPLE = [
                [1.0, 0.3345, 0.4554, 0.4435],
                [1.0, 0.3345, 0.4554, 0.4435],
            ]

            @svc.api(
                route="predict",
                input=bentoml.io.NumpyNdarray.from_sample(INPUT_EXAMPLE),
                output=bentoml.io.NumpyNdarray.from_sample(OUTPUT_EXAMPLE),
            )
            async def predict(req_np_arr: tnp.NDArray) -> tnp.NDArray:

                result: torch.Tensor = await sample_model_runner.async_run(torch.from_numpy(req_np_arr))

                return result.numpy()

                # run server
                # bentoml serve service.py:svc --api-workers=1

    .. tab-item:: BentoML v1.2.x Style

        BentoML v1.2.x higher, there is only one component ``Service``.
        ``Service`` can have inner Service 0 or more than one.
        In this case, ``Runner`` should converted to ``Service`` via ``bentoml.runner_service(...)``

        .. code-block:: python

            # bentoml v1.2.x Style
            # service.py
            from __future__ import annotations

            import bentoml
            import torch
            from numpy import typing as tnp

            SampleModelRunnerInnerService = bentoml.runner_service(
                runner=bentoml.models.get("hello-bento-samplemodel:2024-03-17").to_runner()
            )

            INPUT_EXAMPLE = [
                [1.0, 0.3345, 0.4554, 0.4435],
                [1.0, 0.3345, 0.4554, 0.4435],
            ]

            @bentoml.service(workers=1)  # same to --api-workers=1
            class HelloBentoService:
                sample_model_inner_svc = bentoml.depends(on=SampleModelRunnerInnerService)

                @bentoml.api
                async def predict(
                        self,
                        req_np_arr: Annotated[torch.Tensor, Shape((2, 4)), DType("float32")] = Field(example=INPUT_EXAMPLE)
                ) -> np.ndarray:
                    result: torch.Tensor = await self.sample_model_inner_svc.to_async.__call__(req_np_arr)

                    return result.numpy()

                # run server
                # bentoml serve service.py:HelloBentoService
