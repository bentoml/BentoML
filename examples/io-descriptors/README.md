# BentoML Input/Output Types Tutorial

A BentoML service supports a wide range of data types when creating a Service API. The data types can be catagorized as follows:
- Python Standards: `str`, `int`, `float`, `list`, `dict` etc.
- Pydantic field types: see [Pydantic types documentation](https://field-idempotency--pydantic-docs.netlify.app/usage/types/).
- ML specific types: `nummpy.ndarray`, `torch.Tensor` , `tf.Tensor` for tensor data, `pd.DataFrame` for tabular data, `PIL.Image.Image` for
Image data, and `pathlib.Path` for files such as audios, images, and pdfs.

When creating a Bentoml service, you should use Python's type annotations to define the expected input and output types for each API endpoint. This
step can not only helps validate the data against the specified schema, but also enhances the clarity and readability of your code. Type annotations play
an important role in generating the BentoML API, client, and service UI components, ensuring a consitent and predictable interaction with the Service.

You can also use `pydantic.Field` to set additional information about servicee parameters, such as default values and descriptions. This improves the API's
usability and provides basic documentation. See the following examples for details.

## Running this example

Let's start with the environment. We recommend using virtual environment for better package handling.

```bash
python -m venv io-descriptors-example
source io-descriptors-example/bin/activate
pip install -r requirements.txt
```

7 different API services are implemented within this example, with diversed input/output types. When running, you should specified the class name of the service
you'd like to run inside `bentofile.yaml`.

```yaml
service: "service.py:AudioSpeedUp"
include:
  - "service.py"
```

In the above configuration through `bentofile.yaml`, we're running the `AudioSpeedUp` service, which you can find on line 62 of `service.py`. When running a different
service, simply replace `AudioSpeedUp` with the class name of the service.

For example, if you want to run the first service `ImageResize`, you can configure the `bentofile.yaml` as follows:

```yaml
service: "service.py:ImageResize"
include:
  - "service.py"
```

After you finished configuring `bentofile.yaml`, run `bentoml serve .` to deploy the service locally. You can then interact with the auto-generated swagger UI to play
around with each different API endpoints.

## Different data types

### Standard Python types

The following demonstrates a simple addtion service, with both inputs and output as float parameters. You can
obviously change the type annotation to `int`, `str` etc. to get familiar with the interaction between type
annotaions and the auto-generated Swagger UI when deploying locally.\

```python
@bentoml.service()
class AdditionService:

    @bentoml.api()
    def add(self, num1: float, num2: float) -> float:
        return num1 + num2
```

### Files

Files are handled through `pathlib.Path` in BentoML (which means you should handle the file as a file path in your API implementation as well as on the client side).
Most file types can be specified through `bentoml.validators.Contentype(<file_type>)`. The input of this function follows the standard of the
request format (such as `text/plain`, `application/pdf`, `audio/mp3` etc.).

##### Appending Strings to File example
```python
@bentoml.service()
class AppendStringToFile:

    @bentoml.api()
    def append_string_to_eof(
        self,
        txt_file: t.Annotated[Path, bentoml.validators.ContentType("text/plain")], input_string: str
        ) -> t.Annotated[Path, bentoml.validators.ContentType("text/plain")]:
        with open(txt_file, "a") as file:
            file.write(input_string)
        return txt_file
```

Within `service.py`, example API services with 4 different file types are implemented (audio, image, text file, and pdf file). The functionality of each service
is quite simple and self-explainatory.

Notice that for class `ImageResize`, two different API endpoints are implemented. This is because BentoML can support images parameters directly through
`PIL.Image.Image`, which means that image objects can be directly passed through clients, instead of a file object.

The last two services are examples of having `numpy.ndarray` or `pandas.DataFrame` as input parameters. Since they all work quite similarly with the above examples,
we will not specifically explain them in this tutorial. You can try to write revise the service with `torch.Tensor` as input to check your understanding.

To serve the these examples locally, run `bentoml serve .`

```bash
bentoml serve .
2024-03-22T19:25:24+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:ImageResize" listening on http://localhost:3000 (Press CTRL+C to quit)
```

Open your web browser at http://0.0.0.0:3000 to view the Swagger UI for sending test requests.

You may also send request with `curl` command or any HTTP client, e.g.:

```bash
curl -X 'POST' \
  'http://0.0.0.0:3000/classify_generated_texts' \
  -H 'accept: application/json' \
  -H 'Content-Type: text/plain' \
  -d 'I have an idea!'
```

## Deploy to BentoCloud
Run the following command to deploy this example to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.
```bash
bentoml deploy .
```
For more information, see [Create Deployments](https://docs.bentoml.com/en/latest/bentocloud/how-tos/create-deployments.html).
