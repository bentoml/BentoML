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

.. note::

    ``bentoml.diffusers.save_model`` can also be used to import diffusion models from Hugging Face, but it requires models to be loaded into memory first,
    which is resource-intensive for large models. By contrast, ``bentoml.diffusers.import_model`` tries to import diffusion models directly without loading them into memory.

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
    The ``signatures`` argument of type :ref:`concepts/model:Model signatures` in :obj:`bentoml.diffusers.import_model` is used to determine which methods will be used for inference and exposed in the Runner. The signatures dictionary will then be used during the creation process of a Runner instance.

The signatures used for creating a Runner is ``{"__call__": {"batchable": False}}``. This means by default, BentoML's :doc:`/guides/batching` is disabled when using :obj:`~bentoml.diffusers.import_model()`. If you want to utilize adaptive batching behavior and know your model's dynamic batching dimension, make sure to pass in ``signatures`` as follow:


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
	res = await sd21_runner.async_run(**input_data)
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


Inference with Fine-tuned Models
--------------------------------

LoRA (Low-Rank Adaptation) and textual inversion are 2 methods of fine-tuning a diffusion model. :code:`bentoml.diffusers` currently support them in inference.


Using LoRA
==========

You can load a LoRA layer saved on local disk when creating the runner:

.. code-block:: python
    :caption: `service.py`

    import bentoml
    from bentoml.io import Image, JSON

    bento_model = bentoml.diffusers.get("sd2.1:latest")
    sd21_runner = bento_model.to_runner(name="sd21-runner")

    stable_diffusion_runner = bento_model.with_options(
	pipeline_class=diffusers.StableDiffusionPipeline,
	lora_weights="light_and_shadow.safetensors",
    ).to_runner()

    ...


A runner will only allow loading LoRA weights if it's pipeline's class is a subclass of :code:`diffusers.LoraLoaderMixin`. That's why we specify :code:`pipeline_class=diffusers.StableDiffusionPipeline` in codes above. For more complex LoRA weight loading, you can pass a dictionary instead of a string. For example, if you want to use the LoRA weight stored in file :code:`light_and_shadow.safetensors` hosted at https://huggingface.co/sayakpaul/civitai-light-shadow-lora, you can provide the :code:`lora_weights` option like the following:

.. code-block:: python

    sd21_runner = bento_model.with_options(
	pipeline_class=diffusers.StableDiffusionPipeline,
	textual_inversions="easynegative.safetensors",
	lora_weights=dict(model_name="sayakpaul/civitai-light-shadow-lora", weight_name="light_and_shadow.safetensors"),
    ).to_runner()

For a runner of which the pipeline's class is a subclass of :code:`diffusers.LoraLoaderMixin`, you can also dynamically applying a LoRA weight by calling the runner with an extra parameter :code:`lora_weights`. The codes below will randomly choose a LoRA weight file to be applied to the current generation process. After the image is generated, the LoRA weight will be unloaded from the pipeline

.. code-block:: python
    :caption: `service.py`

    import random
    import bentoml
    from bentoml.io import Image, JSON

    bento_model = bentoml.diffusers.get("sd2.1:latest")
    sd21_runner = bento_model.with_options(
	pipeline_class=diffusers.StableDiffusionPipeline,
    ).to_runner()

    svc = bentoml.Service("stable-diffusion-21", runners=[sd21_runner])

    @svc.api(input=JSON(), output=Image())
    async def txt2img(input_data):
        weights = ["lora1.safetensors", "lora2.safetensors"]
	weight_name = random.choice(weights)
	input_data["lora_weights"] = weight_name
	res = await sd21_runner.async_run(**input_data)
	images = res[0]
	return images[0]


Using Textual Inversion
=======================

Using textual inversion is very similar to using LoRA. You can load a textual inversion saved on local disk when creating the runner:

.. code-block:: python
    :caption: `service.py`

    import bentoml
    from bentoml.io import Image, JSON

    bento_model = bentoml.diffusers.get("sd2.1:latest")
    sd21_runner = bento_model.to_runner(name="sd21-runner")

    stable_diffusion_runner = bento_model.with_options(
	pipeline_class=diffusers.StableDiffusionPipeline,
	textual_inversions="easynegative.safetensors",
    ).to_runner()

    ...

However, you cannot load textual inversion dynamically like LoRA currently.

.. currentmodule:: bentoml.diffusers
