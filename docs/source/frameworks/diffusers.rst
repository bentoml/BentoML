=========
Diffusers
=========

BentoML provides native support for serving and deploying diffusion models utilizing huggingface's `diffusers <https://github.com/huggingface/diffusers>`_ library. Some of the arguments of :code:`bentoml.diffusers` mirrors corresponding arguments of huggingface's diffusers. To get more information about diffusers, please visit `diffusers's official documentation <https://huggingface.co/docs/diffusers/index>`_

Importing a Pre-trained Model
-----------------------------

You can import a pretrained diffusion model from huggingface hub or local directory:

.. code-block:: python
    :caption: `import_model.py`

    import bentoml

    bentoml.diffusers.import_model(
	"sd2.1",  # model tag in BentoML model store
	"stabilityai/stable-diffusion-2-1",  # huggingface model name
    )


If you plan to use the model with a custom pipeline that has method other than :code:`__call__` (e.g. a :code:`StableDiffusionMegaPipeline`), you need to explicitly list them like this:

.. code-block:: python
    :caption: `import_model_mega.py`

    import bentoml

    bentoml.diffusers.import_model(
	"sd2",
	"stabilityai/stable-diffusion-2",
	signatures={
	    "__call__": {
		"batchable": False
	    },
	    "text2img": {
		"batchable": False
	    },
	    "img2img": {
		"batchable": False
	    },
	    "inpaint": {
		"batchable": False
	    },
	}
    )


.. note::

    :code:`bentoml.diffusers.import_model` has parameter ``signatures``.
    The ``signatures`` argument of type :ref:`Model Signatures <concepts/model:Model Signatures>` in :obj:`bentoml.diffusers.import_model` is used to determine which methods will be used for inference and exposed in the Runner. The signatures dictionary will then be used during the creation process of a Runner instance.

The signatures used for creating a Runner is ``{"__call__": {"batchable": False}}``. This means by default, BentoML’s `Adaptive Batching <guides/batching:Adaptive Batching>`_ is disabled when using :obj:`~bentoml.diffusers.import_model()`. If you want to utilize adaptive batching behavior and know your model's dynamic batching dimension, make sure to pass in ``signatures`` as follow:


.. code-block:: python

    bentoml.diffusers.import_model(model_name, "my_model", signatures={"__call__": {"batch_dim": 0, "batchable": True}})


Building a Service
------------------

Create a BentoML service with the previously saved :code:`sd2.1` model using the :code:`bentoml.diffusers` framework APIs.

.. code-block:: python
    :caption: `service.py`

    import bentoml
    from bentoml.io import Image, JSON

    bento_model = bentoml.diffusers.get("sd2.1:latest")
    sd21_runner = bento_model.to_runner(name="sd21-runner")

    svc = bentoml.Service("stable-diffusion-21", runners=[sd21_runner])

    @svc.api(input=JSON(), output=Image())
    async def txt2img(input_data):
	kwargs = input_data.dict()
	res = await sd21_runner.async_run(**kwargs)
	images = res[0]
	return images[0]

.. note::

   the :code:`bentoml.diffusers`'s runner will return a tuple instead of :code:`diffusers.utils.BaseOutput`. The first element of this tuple is usually what you want. :code`bentoml.diffusers` do this to make sure the returned value from runner can be pickled and transferred to remote host, so that distributed deployment can be achieved.

When creating the runner, you can provide a pipeline class (instead of the default :code:`StableDiffusionPipeline`) and/or a custome pipeline name to get different features from the diffusion model. For example, below is an example of using :code:`StableDiffusionMegaPipeline` to have both :code:`txt2img` and :code:`img2img` in the same runner and service:

.. code-block:: python
    :caption: `service_mega.py`

    from diffusers import DiffusionPipeline

    import bentoml
    from bentoml.io import Image, JSON, Multipart

    bento_model = bentoml.diffusers.get("sd2:latest")
    stable_diffusion_runner = bento_model.with_options(
	pipeline_class=DiffusionPipeline,
	custom_pipeline="stable_diffusion_mega",
    ).to_runner()

    svc = bentoml.Service("stable_diffusion_v2_mega", runners=[stable_diffusion_runner])

    @svc.api(input=JSON(), output=Image())
    def txt2img(input_data):
	res = stable_diffusion_runner.text2img.run(**input_data)
	images = res[0]
	return images[0]

    img2img_input_spec = Multipart(img=Image(), data=JSON())
    @svc.api(input=img2img_input_spec, output=Image())
    def img2img(img, data):
	data["image"] = img
	res = stable_diffusion_runner.img2img.run(**data)
	images = res[0]
	return images[0]


.. currentmodule:: bentoml.diffusers
