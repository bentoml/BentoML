Detectron2
----------

| Detectron2 is a platform for object detection, segmentation and other visual
| recognition tasks. - `Source <https://github.com/facebookresearch/detectron2>`_

Users can now use Detectron2 with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

    import numpy as np
    import torch
    import pytest
    import imageio
    from detectron2 import model_zoo
    from detectron2.data import transforms as T
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model

    import bentoml

    def detectron_model_and_config() -> t.Tuple[torch.nn.Module, "CfgNode"]:
        model_url: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

        cfg: "CfgNode" = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_url))
        # set threshold for this model
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url)

        cloned = cfg.clone()
        cloned.MODEL.DEVICE = "cpu"  # running on CI
        model: torch.nn.Module = build_model(cloned)
        model.eval()

        return model, cfg

    # `save` a given classifier and retrieve coresponding tag:
    model, config = detectron_model_and_config()
    tag_info = bentoml.detectron.save(
        "mask_rcnn_R_50_FPN_3x",
        model,
        model_config=config,
    )

    # retrieve metadata with `bentoml.models.get`:
    metadata = bentoml.models.get(tag)

    # `load` the model back in memory:
    model = bentoml.detectron2.load("mask_rcnn_R_50_FPN_3x")

    def prepare_image(
        original_image: "np.ndarray[t.Any, np.dtype[t.Any]]",
    ) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
        """Mainly to test on COCO dataset"""
        _aug = T.ResizeShortestEdge([800, 800], 1333)

        image = _aug.get_transform(original_image).apply_image(original_image)
        return image.transpose(2, 0, 1)

    # Run a given model under `Runner` abstraction with `load_runner`
    runner = bentoml.detectron2.load_runner(tag)
    image = torch.as_tensor(prepare_image(np.asarray(imageio.imread(IMAGE_URL))))
    res = runner.run_batch(image)
    runner.run_batch(imarr)

.. admonition:: btw
   :class: customNotesFmt

   You can find more examples for **Detectron2** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.detectron

.. autofunction:: bentoml.detectron.save

.. autofunction:: bentoml.detectron.load

.. autofunction:: bentoml.detectron.load_runner
