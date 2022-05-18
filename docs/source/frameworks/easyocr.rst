=======
EasyOCR
=======

Users can now use EasyOCR with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

    import bentoml
    import easyocr
    import numpy as np
    import imageio

    IMAGE_PATH: str = "./tests/utils/_static/chinese.jpg"
    LANG_LIST = ["ch_sim", "en"]
    RECOG_NETWORK = "zh_sim_g2"
    DETECT_MODEL = "craft_mlt_25k"

    model = easyocr.Reader(
        lang_list=LANG_LIST,
        gpu=False,
        download_enabled=True,
        recog_network=RECOG_NETWORK,
    )

    # `save` a given classifier and retrieve coresponding tag:
    tag = bentoml.easyocr.save(
        "ocr_craft",
        model,
        lang_list=LANG_LIST,
        recog_network=RECOG_NETWORK,
        detect_model=DETECT_MODEL,
    )

    # retrieve metadata with `bentoml.models.get`:
    metadata = bentoml.models.get(tag)

    # `load` the model back in memory:
    model = bentoml.easyocr.load("ocr_craft")

    # Run a given model under `Runner` abstraction with `load_runner`
    imarr = np.asarray(imageio.imread(IMAGE_PATH))
    runner = bentoml.easyocr.load_runner(tag)
    runner.run_batch(imarr)

.. note::

   You can find more examples for **EasyOCR** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.easyocr

.. autofunction:: bentoml.easyocr.save

.. autofunction:: bentoml.easyocr.load

.. autofunction:: bentoml.easyocr.load_runner
