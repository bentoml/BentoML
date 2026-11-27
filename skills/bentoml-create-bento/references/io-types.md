# Input and output types

The annotations on an `@bentoml.api` method are the request schema, the response schema, the
OpenAPI spec and the generated client. Everything here is enforced by pydantic v2.

## Supported types

| Type | Parameter | Wire format in | Return | Wire format out |
|---|---|---|---|---|
| Scalars, `list`, `dict` | `text: str, n: int` | `application/json` object keyed by parameter name | `-> str`, `-> bytes` | `text/plain` |
| | | | `-> int/float/dict/list` | `application/json` |
| Pydantic model | `params: MyModel` | nested JSON object under the parameter name | `-> MyModel` | JSON |
| `numpy.ndarray`, `torch.Tensor`, `tensorflow.Tensor` | `x: np.ndarray` | JSON nested lists | same | JSON nested lists |
| `pandas.DataFrame` | `df: pd.DataFrame` | JSON records or columns | same | JSON records |
| `PIL.Image.Image` | `img: Image` | `multipart/form-data` (file or URL) | same | `image/<detected>` |
| `pathlib.Path` | `f: Path` | `multipart/form-data` (file or URL) | same | binary, MIME from suffix |
| `Generator` / `AsyncGenerator` | — | — | `-> Generator[str, None, None]` | `text/event-stream` |
| `list[...]` of the above | `imgs: list[Image]` | multipart | `-> list[dict]` | JSON |

`bentoml.Context` (as `ctx: bentoml.Context` or `context:`) is not part of the schema.

The **return** annotation is enforced too, not just documented: annotating `-> list[dict[str,
float]]` turns an `int` row index into `0.0` on the wire. Use `dict[str, t.Any]`, or a pydantic
model, when a field's type is not uniform.

Validation failures return **400** with a pydantic `detail` array naming the offending field —
not 422. A type the validator cannot even attempt (a string where a tensor is expected) comes
back as a 500.

Not supported: general unions (`int | str`); a response mixing raw binary with JSON fields;
multiple binary values in one response. Return a `dict` of JSON plus a URL, or split the API.

## Defaults, examples, descriptions

```python
from pydantic import Field

@bentoml.api
def generate(
    self,
    prompt: str = Field(description="The prompt text"),
    temperature: float = Field(default=0.7, description="0-2"),
    stop: list[str] | None = Field(default=None),          # Optional MUST have a default
) -> str: ...
```

`Field(examples=[[0.1, 0.4]])` prefills the Swagger UI and is the cheapest way to make an
endpoint self-documenting. A plain default value (`text: str = EXAMPLE_INPUT`) does the same.

## Validators

`bentoml.validators` adds ML-specific constraints; standard `annotated_types` (`Ge`, `Lt`,
`MaxLen`, `MultipleOf`) and every pydantic annotation also work.

```python
from typing import Annotated
from bentoml.validators import ContentType, DType, DataframeSchema, Shape

audio: Annotated[Path, ContentType("audio/mp3")]
x:     Annotated[torch.Tensor, Shape((1, 4)), DType("float32")]
df:    Annotated[pd.DataFrame, DataframeSchema(orient="records", columns=["a", "b"])]
prompt: Annotated[str, MaxLen(1000)]
```

| Validator | Applies to |
|---|---|
| `ContentType` | `Path`, `bytes`, `PIL.Image.Image` |

| `Shape`, `DType` | `numpy.ndarray`, `torch.Tensor`, `tensorflow.Tensor` |
| `DataframeSchema(orient="records"\|"columns", columns=[...])` | `pandas.DataFrame` |

`ContentType` checks the MIME type **the client declares**, not the bytes. `curl -F
"csv=@x.csv"` declares `application/octet-stream` and gets a 400 — the working form is
`curl -F "csv=@x.csv;type=text/csv"`, and in Python
`files={"csv": ("x.csv", data, "text/csv")}`. Annotate a content type only if you want that
enforced; a bare `Path` accepts anything.

## Pydantic models

A model as a parameter type nests the payload under that parameter's name. To accept the
model's fields at the top level of the request body instead, use `input_spec`:

```python
@bentoml.api(input_spec=GenerationParams)
def generate(self, **params: t.Any) -> str:
    return self.llm(params["prompt"], temperature=params["temperature"])
```

`pydantic.BaseModel` fields only accept built-in types. For a model with `numpy.ndarray`,
`pandas.DataFrame` or tensor fields, subclass `bentoml.IODescriptor` instead:

```python
class MyInput(bentoml.IODescriptor):
    data: np.ndarray[tuple[int], np.dtype[np.float16]]
```

## Files

In: annotate `pathlib.Path`; BentoML writes the upload to a temp file and hands you the path.
Out: write into the per-request `ctx.temp_dir` and return the `Path`.

```python
@bentoml.api
def to_speech(self, text: str, ctx: bentoml.Context) -> Annotated[Path, ContentType("audio/mp3")]:
    out = Path(ctx.temp_dir) / "out.mp3"
    out.write_bytes(self.tts(text))
    return out
```

Returning `bytes` with a `ContentType` annotation avoids touching disk at all.

## Root input

One positional-only parameter (before `/` in the signature) takes the whole request body with
no JSON wrapper — the natural shape for raw image/audio/text uploads. At most one, and no other
parameters except the context.

```python
@bentoml.api
def upload(self, image: PILImage.Image, /) -> dict: ...
# curl -X POST --data-binary @cat.png localhost:3000/upload
# client.upload(path)          <- positional; client.upload(image=path) is wrong
```

## Streaming

Return a generator. The response is `text/event-stream`; yield strings (or bytes).

```python
@bentoml.api
async def chat(self, prompt: str) -> AsyncGenerator[str, None]:
    async for chunk in self.engine.stream(prompt):
        yield chunk
```

A `@bentoml.task` cannot stream.

## Batching

`@bentoml.api(batchable=True)` turns on server-side adaptive batching: the endpoint receives a
*list* of what individual callers sent, and must return a list of the same length in the same
order.

```python
@bentoml.api(batchable=True, max_batch_size=32, max_latency_ms=1000)
def encode(self, sentences: list[str]) -> np.ndarray: ...
```

- Exactly one parameter besides the context. For several, wrap them in a pydantic model and
  take `list[Model]`; a thin non-batchable wrapper Service gives clients the per-request shape.
- `batch_dim` controls which axis is concatenated for arrays (default 0).
- Exceeding `max_latency_ms` returns 503 to the caller.
- A **sync** caller in another Service sends one request at a time (`threads=1`), so batches
  never form: use `async` + `.to_async`, or raise `threads` on the caller.

Full reference: https://docs.bentoml.com/en/latest/build-with-bentoml/iotypes.html
