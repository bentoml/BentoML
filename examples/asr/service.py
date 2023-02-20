from __future__ import annotations

import typing as t
from os import path
from os import environ

from whispercpp.utils import LazyLoader

import bentoml

if t.TYPE_CHECKING:
    import numpy as np
    import whispercpp as w
    from numpy.typing import NDArray
else:
    np = LazyLoader("np", globals(), "numpy")
    w = LazyLoader("w", globals(), "whispercpp")


class WhisperCppRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("mps", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        model_name = environ.get("GGML_MODEL", "tiny.en")
        self.model = w.Whisper.from_pretrained(model_name)

    @bentoml.Runnable.method(batchable=True, batch_dim=(0, 0))
    def transcribe_array(self, arr: NDArray[np.float32]):
        return self.model.transcribe(arr)

    @bentoml.Runnable.method(batchable=True, batch_dim=(0, 0))
    def transcribe_file(self, p: str):
        resolved = path.expanduser(path.abspath(p))
        if not path.exists(resolved):
            raise FileNotFoundError(resolved)
        return self.model.transcribe_from_file(resolved)

    @bentoml.Runnable.method(batchable=False)
    def stream(self):
        self.model.stream_transcribe()


cpp_runner = bentoml.Runner(WhisperCppRunnable, max_batch_size=30)

svc = bentoml.Service("whispercpp_asr", runners=[cpp_runner])


@svc.api(
    input=bentoml.io.Text.from_sample("./samples/jfk.wav"),
    output=bentoml.io.Text(),
)
async def transcribe_file(input_file: str):
    return await cpp_runner.transcribe_file.async_run(input_file)


@svc.api(input=bentoml.io.NumpyNdarray(), output=bentoml.io.Text())
async def transcribe_ndarray(arr: NDArray[np.float32]):
    return await cpp_runner.transcribe_array.async_run(arr)
