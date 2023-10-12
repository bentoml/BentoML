==========
Detectron2
==========

`Detectron2 <https://github.com/facebookresearch/detectron2>`_ is Facebook AI Research's (FAIR) next generation library
that provides state-of-the-art detection and segmentation algorithms. It is the successor of Detectron and maskrcnn-benchmark.
It supports a number of computer vision research projects and production applications in Facebook.

This guide will provide an overview of how to save and load Detectron2 models with BentoML.

Compatibility
-------------

Since the official way to install Detectron2 from their `installation guide <https://detectron2.readthedocs.io/en/latest/tutorials/install.html>`_ is from the repository git source,
The current implementation has been tested with Detectron2 since this :github:`commit <facebookresearch/detectron2/commit/4e447553eb32b6e3784df0b8fca286935107b2fd>`

Saving a Detectron2 model
-------------------------

The following example is excerpted from `Detectron2's official demo <https://github.com/facebookresearch/detectron2/blob/main/demo/demo.py>`_.

We will be using a Masked-RCNN trained on the COCO datasets.

.. code-block:: python

   import detectron2.config as Cf
   import detectron2.model_zoo as ModelZoo
   import detectron2.modeling as Modeling

   cfg = Cf.get_cfg()
   cfg.merge_from_file(ModelZoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
   cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
   cfg.MODEL.WEIGHTS = ModelZoo.get_checkpoint_url(model_url)
   cfg = cfg.clone()
   cfg.MODEL.DEVICE = "cpu"  # Change device to run on CPU
   model = Modeling.build_model(cfg)
   model.eval()

To run an example inputs, use the following image from the COCO dataset:

.. code-block:: python

   import requests
   import PIL.Image
   import numpy as np
   from detectron2.data import transforms as Transforms

   url = "http://images.cocodataset.org/val2017/000000439715.jpg"
   augmentation = Transforms.ResizeShortestEdge([800,800], 1333)

   im = np.asarray(PIL.Image.open(requests.get(url, stream=True).raw).convert("RGB"))
   arr = augmentation.get_transform(im).apply_image(im)

   # Running prediction
   outputs = model({'image': torch.as_tensor(arr.transpose(2, 0, 1))})

To save the above model, pass in both the model and the config instance to :obj:`~bentoml.detectron.save_model`:

.. code-block:: python

   bentoml.detectron.save_model("coco-masked-rcnn", model, config=cfg)


Detectron2 also provides a :obj:`~detectron2.engine.defaults.DefaultPredictor` class that can be used to run inference on the model quickly.

BentoML also supports saving this predictor with ``bentoml.detectron.save_model``. Do the following:

.. code-block:: python

   from detectron2.engine import DefaultPredictor

   predictor = DefaultPredictor(cfg)
   bentoml.detectron.save_model("coco-masked-rcnn-predictor", predictor)


.. note::

    :bdg-info:`Remarks:` External python classes or utility functions required by the Detectron models/custom models
    must be referenced in ``<module>.<class>`` format, and such modules should be passed to ``bentoml.detectron.save_model`` via ``external_modules``.

    For example:

    .. code-block:: python

       import dit

       predictor = dit.get_predictor()

       bentoml.detectron.save_model("dit-predictor", predictor, external_modules=[dit])

    This is due to a limitation from PyTorch model serialisation, where PyTorch requires the model's source code to restore it.

The signatures used for creating a Runner is ``{"__call__": {"batchable": False}}``. This means by default, BentoML's :doc:`/guides/batching` is disabled when using :obj:`~bentoml.pytorch.save_model()`. If you want to utilize adaptive batching behavior and know your model's dynamic batching dimension, make sure to pass in ``signatures`` as follow:

.. code-block:: python

   bentoml.detectron.save_model("dit-predictor", predictor, signatures={"__call__": {"batch_dim": 0, "batchable": True}})

.. admonition:: About ``custom_objects``

   BentoML will save the Detectron ``config`` ConfigNode instance into the ``custom_objects`` attribute via ``config``. Make sure when using ``custom_objects`` to not overwrite this naming.


Building a Service
------------------

Create a BentoML service with the previously saved ``coco-masked-rcnn-predictor`` model using the :code:`bentoml.detectron` framework APIs.

.. code-block:: python
    :caption: `service.py`

    import bentoml
    import PIL.Image
    import numpy as np
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import ColorMode
    from detectron2.utils.visualizer import Visualizer

    bentomodel = bentoml.detectron.get("coco-masked-rcnn-predictor")
    cfg = model.custom_objects['config']
    runner = bentomodel.to_runner()

    svc = bentoml.Service(name="masked-rcnn", runners=[runner])

    @svc.api(input=bentoml.io.Image(), output=bentoml.io.Image())
    async def predict(im: PIL.Image.Image) -> PIL.Image.Image:
        md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        tensor = np.array(im)
        output = await predictor.async_run(tensor)
        instances = output['instances']
        v = Visualizer(
            tensor[:, :, ::-1], md, scale=1.0, instance_mode=ColorMode.SEGMENTATION
        )
        res = v.draw_instance_predictions(output.to("cpu"))
        return Image.fromarray(res.get_image()[:, :, ::-1])


Adaptive Batching
-----------------

Most Detectron models can accept batched data as input. If batched interence is supported, it is recommended to enable batching to take advantage of
the adaptive batching capability to improve the throughput and efficiency of the model. Enable adaptive batching by overriding the :code:`signatures`
argument with the method name and providing :code:`batchable` and :code:`batch_dim` configurations when saving the model to the model store.

.. seealso::

   See :ref:`Adaptive Batching <guides/batching:Adaptive Batching>` to learn more.


.. note::

   You can find more examples for **Detectron** in our :examples:`bentoml/examples <>` directory.

.. currentmodule:: bentoml.detectron
