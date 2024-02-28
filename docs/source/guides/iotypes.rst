======================
Input and output types
======================

When creating a BentoML :doc:`Service </guides/services>`, you need to specify the input and output (IO) types the Service's API. These types are integral in shaping the logic of the Service API, guiding the flow of data into and out of the Service. BentoML supports a wide range of data types that are commonly used in Python, Pydantic, as well as types specific to machine learning (ML) workflows. This ensures that BentoML Services can seamlessly integrate with different data sources and ML frameworks.

This document provides a comprehensive overview of the supported Service API schemas in BentoML, with code examples to illustrate their implementation.

Overview
--------

Supported input and output types in BentoML include:

- **Standard Python types**: Basic types like ``str``, ``int``, ``float``, ``boolean``, ``list``, and ``dict``.
- **Pydantic field types**: BentoML extends its type support to `Pydantic field types <https://field-idempotency--pydantic-docs.netlify.app/usage/types/>`_, offering a more structured and validated approach to handling complex data schemas.
- **ML specific types**: To meet the requirements of different ML use cases, BentoML supports types like ``numpy.ndarray``, ``torch.Tensor``, and ``tf.Tensor`` for handling tensor data, ``pd.DataFrame`` for working with tabular data, ``PIL.Image.Image`` for image data, and ``pathlib.Path`` for file path references.

You use Python's type annotations to define the expected input and output types for each API endpoint. This not only helps validate the data against the specified schema but also enhances the clarity and readability of the code. The type annotations play an important role in generating the API, BentoML :doc:`client </guides/clients>`, and UI components, ensuring a consistent and predictable interaction with the Service.

In addition, you can use ``pydantic.Field`` to set additional information about parameters, such as default values and descriptions. This improves the API's usability and provides basic documentation. See the following examples for details.

Define API schemas
------------------

This section presents examples of different API schemas supported in BentoML. Each example shows how to define input and output types for a specific use case.

Standard Python types
^^^^^^^^^^^^^^^^^^^^^

Python's standard types such as strings, integers, floats, booleans, lists, and dictionaries are commonly used for simple data structures. You can easily integrate these types into your Service. Here is an example:

.. code-block:: python

    from pydantic import Field

    @bentoml.service
    class LanguageModel:
        @bentoml.api
        def generate(
            self, prompt: int = Field(description="The prompt text"),
            temperature: float = Field(default=0.0, description="A sampling temperature between 0 and 2"),
            max_tokens: int = Field(default=1000, description="max tokens to use"),
        ) -> Generator[str, None, None]:
            # Implementation of the language generation model
            ...

This example uses Python's type annotations (for example, ``str``, ``float``, and ``int``) to specify the expected data type for each parameter. It returns a generator, which can stream the response. ``pydantic.Field`` can be used to set default values and provide descriptions for parameters.

Example and nullable input
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can define APIs that accept inputs with examples or nullable fields.

To set an example value, you can use ``pydantic.Field``:

.. code-block:: python

    from pydantic import Field

    @bentoml.service
    class IrisClassifier:
        @bentoml.api
        def classify(self, input: np.ndarray = Field(examples=[[0.1, 0.4, 0.2, 1.0]]) -> np.ndarray:
            ...

To handle nullable input, you can use ``Optional``:

.. code-block:: python

    from pydantic import Field
    from typing import Optional

    @bentoml.service
    class LanguageModel:
        @bentoml.api
        def generate(
            self, prompt: int = Field(description="The prompt text"),
            temperature: Optional[float] = Field(default=None, description="A sampling temperature between 0 and 2"),
            max_tokens: Optional[float] = Field(default=None, description="max tokens to use"),
        ) -> Generator[str, None, None]:
            ...

In the ``LanguageModel`` class, the ``temperature`` and ``max_tokens`` fields are marked as ``Optional``. This means they can be ``None``. Note that when using ``Optional`` types in BentoML, you must provide a default value (here, ``default=None``). General union types are not supported.

Pydantic
^^^^^^^^

Pydantic models allow for more structured data with validation. They are particularly useful when your Service needs to handle complex data structures with rigorous validation requirements. Here is an example:

.. code-block:: python

    from pydantic import BaseModel, Field

    # Define a Pydantic model for structured data input
    class AdsGenerationParams(BaseModel):
        prompt: str = Field(description="The prompt text")
        industry: str = Field(description="The industry the company belongs to")
        target_audience: str = Field(description="Target audience for the advertisement")
        temperature: float = Field(default=0.0, description="A sampling temperature between 0 and 2")

    @bentoml.service
    class AdsWriter:
        @bentoml.api
        def generate(self, params: AdsGenerationParams) -> str:
            # Implementation logic
            ...

In the above code snippet, the ``AdsGenerationParams`` class is a Pydantic model which defines the structure and validation of input data. Each field in the class is annotated with a type, and can include default values and descriptions. Pydantic automatically validates incoming data against the ``AdsGenerationParams`` schema. If the data doesn’t conform to the schema, an error will be raised before the method is executed.

Files
^^^^^

You handle file input and output using ``pathlib.Path``. It is helpful for Services that process files, such as audio, images, and documents.

Here's a simple example that accepts a ``Path`` object as input, representing the path to an audio file.

.. code-block:: python

    from pathlib import Path

    @bentoml.service
    class WhisperX:
        @bentoml.api
        def to_text(self, audio: Path) -> str:
            # Implementation for converting audio files to text
            ...

To restrict the file type to a specific format, such as audio files, you can use the ``ContentType`` validator with the ``Annotated`` type. For example, you can let the API method only accept MP3 audio files:

.. code-block:: python

    from pathlib import Path
    from bentoml.validators import ContentType
    from typing import Annotated  # Python 3.9 or above
    from typing_extensions import Annotated  # Older than 3.9

    @bentoml.service
    class WhisperX:
        @bentoml.api
        def to_text(self, audio: Annotated[Path, ContentType("audio/mp3")]) -> str:
            ...

To output a file with a path, you can use ``context.temp_dir`` to provide a unique temporary directory for each request and store the output file. For example:

.. code-block:: python

    from pathlib import Path

    @bentoml.service
    class Vits:
        @bentoml.api
        def to_speech(self, text: str, context: bentoml.Context) -> Path:
            # Example text-to-speech synthesis implementation
            audio_bytes = self.tts.synthesize(text)
            # Writing the audio bytes to a file in the temporary directory
            with open(Path(context.temp_dir) / "output.mp3", "wb") as f:
                f.write(audio_bytes)
            # Returning the path to the generated audio file directly
            return Path(context.temp_dir) / "output.mp3"

When the method returns a ``Path`` object pointing to the generated file, BentoML serializes this file and includes it in the response to the client.

If you don't want to save temporary files to disk, you can return the data as ``bytes`` instead of ``pathlib.Path`` with properly annotated ``ContentType``. This is efficient for Services that generate data on the fly.

Tensors
^^^^^^^

BentoML supports various tensor types such as ``numpy.ndarray``, ``torch.Tensor``, and ``tf.Tensor``. Additionally, you can use :ref:`reference/sdk:bentoml.validators` like ``bentoml.Shape`` and ``bentoml.DType`` to enforce specific shapes and data types for tensor input. Here is an example:

.. code-block:: python

    import torch
    from bentoml.validators import Shape, Dtype
    from typing import Annotated  # Python 3.9 or above
    from typing_extensions import Annotated  # Older than 3.9
    from pydantic import Field

    @bentoml.service
    class IrisClassifier:
        @bentoml.api
        def classify(
            self,
            input: Annotated[torch.Tensor, Shape((1, 4)), Dtype("float32")]
            = Field(description="A 1x4 tensor with float32 dtype")
        ) -> np.ndarray:
            ...

In this example:

- The ``classify`` method expects ``torch.Tensor`` input.
- The ``Annotated`` type is used with ``Shape`` and ``Dtype`` validators to specify that the expected tensor should have a shape of ``(1, 4)`` and a data type of ``float32``.
- ``pydantic.Field`` provides an additional description for the input parameter for better readability of the API.

Tabular
^^^^^^^

Pandas DataFrames are commonly used for handling tabular data in machine learning. BentoML supports Pandas DataFrame input and allows you to annotate them with validators to ensure the data conforms to the expected structure.

Here is an example:

.. code-block:: python

    from typing import Annotated  # Python 3.9 or above
    from typing_extensions import Annotated  # Older than 3.9
    import pandas as pd
    from bentoml.validators import DataframeSchema

    @bentoml.service
    class IrisClassifier:
        @bentoml.api
        def classify(
            self,
            input: Annotated[pd.Dataframe, DataframeSchema(orient="records", columns=["petal_length", "petal_width"])
        ) -> int:
            # Classification logic using the input DataFrame
            ...

In this example:

- The ``classify`` method of the ``IrisClassifier`` Service accepts a Pandas DataFrame as input.
- The ``Annotated`` type is used with ``DataframeSchema`` to specify the expected orientation and columns of the DataFrame.
    - ``orient="records"`` indicates that the DataFrame is expected in a record-oriented format.
    - ``columns=["petal_length", "petal_width"]`` specifies the expected columns in the DataFrame.

The ``DataframeSchema`` validator supports the following two orientations, which determine how the data is structured when received by the API.

- ``records``: Each row is represented as a dictionary where the keys are column names.
- ``columns``: Data is organized by columns, where each key in the dictionary represents a column, and the corresponding value is a list of column values.

Images
^^^^^^

BentoML Services can work with images through the PIL library or ``pathlib.Path``.

Here is an example of using PIL:

.. code-block:: python

    from PIL.Image import Image as PILImage

    @bentoml.service
    class MnistPredictor:
        @bentoml.api
        def infer(self, input: PILImage) -> int:
            # Image processing and inference logic
            ...

Alternatively, you can use ``pathlib.Path`` with a ``ContentType`` validator to handle image files:

.. code-block:: python

    from pathlib import Path
    from typing import Annotated  # Python 3.9 or above
    from typing_extensions import Annotated  # Older than 3.9
    from bentoml.validators import ContentType

    @bentoml.service
    class MnistPredictor:
        @bentoml.api
        def infer(self, input: Annotated[Path, ContentType('image/jpeg')) -> int:
            ...

This is particularly useful when dealing with image uploads in web applications or similar scenarios.

Validate data
-------------

Proper validation of input data is important for BentoML Services to ensure that the data being processed is in the expected format and meets the necessary quality standards. BentoML provides a simple validation mechanism and supports all the validation features provided by Pydantic by default. This allows for comprehensive checks on the structure, type, and constraints of the input data.

Here is an example:

.. code-block:: python

    from typing import Annotated  # Python 3.9 or above
    from typing_extensions import Annotated  # older than 3.9
    from annotated_types import Ge, Lt, Gt, MultipleOf, MaxLen

    @bentoml.service
    class LLMPredictor:
        @bentoml.api
        def predict(
            self,
            prompt: Annotated[str, MaxLen(1000)],
            temperature: Annotated[float, Ge(0), Lt(2)],
            max_tokens: Annotated[int, Gt(0), MultipleOf(100)]
        ) -> int:
            ...

In this example, the validators ensure that the ``prompt`` string does not exceed 1000 characters, ``temperature`` is between 0 and 2, and ``max_tokens`` is a positive multiple of 100.

Validation for useful ML types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BentoML provides validation capabilities for common ML data types, such as tensors and data frames, to ensure the integrity of the data being fed into models. You can find validation examples for these data types in the above sections.

The following table includes the additional input and output types supported by BentoML, which are specifically designed for ML use cases. The annotations allowed for each type can be used to further refine and validate the data.

.. list-table::
   :header-rows: 1

   * - Type name
     - Description
     - Annotations allowed
   * - ``numpy.ndarray``
     - Multi-dimensional array for numerical data, commonly used in ML tasks.
     - ``bentoml.validators.Shape``, ``bentoml.validators.DType``
   * - ``torch.Tensor``
     - Tensor type in PyTorch for representing tensor data.
     - ``bentoml.validators.Shape``, ``bentoml.validators.DType``
   * - ``tf.Tensor``
     - Tensor type in TensorFlow for representing tensor data.
     - ``bentoml.validators.Shape``, ``bentoml.validators.DType``
   * - ``pd.DataFrame``
     - Data structure for tabular data, commonly used in data analysis.
     - ``bentoml.validators.DataframeSchema``
   * - ``PIL.Image.Image``
     - Image data type from the PIL library, used in image processing.
     - ``bentoml.validators.ContentType``
   * - ``pathlib.Path``
     - File paths, used for file inputs and outputs.
     - ``bentoml.validators.ContentType``
