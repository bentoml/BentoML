Offline Batch Serving
=====================

BentoML CLI allows you to load and run packaged models straight from the CLI without needing to start up a server to serve requests (hence offline batch serving). There are three main classes of input adapters through which you can perform offline batch serving.

:code:`StringInput`
-------------------
:code:`DataframeInput`, :code:`JSONInput`, and :code:`TFTensorInput` all inherit from :code:`StringInput`. This class of adapter should be used on any input that is string-like (e.g. JSON, CSV, regular string, raw bytes).

**Query with CLI command**

Example with :code:`DataframeInput`. Here, we give the data to the input adapter in the form of a flat JSON string using the :code:`--input` flag::

    $ bentoml run IrisClassifier:latest predict --input '[{"sw": 1, "sl": 2, "pw": 1, "pl": 2}]'

You can also pass file data to any subclass of :code:`StringInput` using the :code:`--input-file` flag::

    $ bentoml run IrisClassifier:latest predict --format csv --input-file test.csv


:code:`FileInput`
-----------------
:code:`ImageInput` inherits from :code:`FileInput`. This class of adapter should be used mostly for image data (e.g. JPG, PNG).

**Query with CLI command**
    
Example with :code:`ImageInput`. We provide the image data to the input adapter by specifying the the image file we want to run inference on using the flag :code:`--input-file`::

    $ bentoml run PyTorchFashionClassifier:latest predict --input-file test.jpg

Alternatively, we can also run inference on all images in a folder and specify the batchsize using the flag :code:`--max-match-size`::

    $ bentoml run PyTorchFashionClassifier:latest predict \\
          --input-file folder/*.jpg --max-batch-size 10

:code:`MultiFileInput`
-----------------------
:code:`AnnotatedImageInput` and :code:`MultiImageInput` all inherit from :code:`MultiFileInput`. This class of adapter should be mostly used for models that require multiple images (e.g. models that require a depth-map along with a regular image).

**Query with CLI command**

Example with :code:`MultiImageInput`. We provide image data to the input adapter using CLI flags in the form :code:`--input-<name>` or :code:`--input-file-<name>`::

    $ bentoml run PyTorchFashionClassifier:latest predict \\
          --input-file-imageX testx.jpg \\
          --input-file-imageY testy.jpg

Similarly to :code:`FileInput`, we can infer all file pairs under a folder with ten pairs each batch by specifying the :code:`--max-batch-size` flag::

    $ bentoml run PyTorchFashionClassifier:latest predict --max-batch-size 10 \\
          --input-file-imageX folderx/*.jpg \\
          --input-file-imageY foldery/*.jpg

Note: when running inference using the MultiFileInput under a folder, ensure that the file names have the same prefix. For example::

    folderx:
        - 1.jpg
        - 2.jpg
        ...
    foldery:
        - 1.jpg
        - 2.jpg