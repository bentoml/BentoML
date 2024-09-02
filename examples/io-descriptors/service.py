import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image as im
from PIL.Image import Image
from pydantic import Field

import bentoml
from bentoml.validators import DataframeSchema
from bentoml.validators import DType


@bentoml.service()
class ImageResize:
    @bentoml.api()
    def generate(self, image: Image, height: int = 64, width: int = 64) -> Image:
        size = height, width
        return image.resize(size, im.LANCZOS)

    @bentoml.api()
    def generate_with_path(
        self,
        image: t.Annotated[Path, bentoml.validators.ContentType("image/jpeg")],
        height: int = 64,
        width: int = 64,
    ) -> Image:
        size = height, width
        image = im.open(image)
        return image.resize(size, im.LANCZOS)


@bentoml.service()
class AdditionService:
    @bentoml.api()
    def add(self, num1: float, num2: float) -> float:
        return num1 + num2


@bentoml.service()
class AppendStringToFile:
    @bentoml.api()
    def append_string_to_eof(
        self,
        context: bentoml.Context,
        txt_file: t.Annotated[Path, bentoml.validators.ContentType("text/plain")],
        input_string: str,
    ) -> t.Annotated[Path, bentoml.validators.ContentType("text/plain")]:
        with open(txt_file, "a") as file:
            file.write(input_string)
        return txt_file


@bentoml.service()
class PDFtoImage:
    @bentoml.api()
    def pdf_first_page_as_image(
        self,
        pdf: t.Annotated[Path, bentoml.validators.ContentType("application/pdf")],
    ) -> Image:
        from pdf2image import convert_from_path

        pages = convert_from_path(pdf)
        return pages[0].resize(pages[0].size, im.ANTIALIAS)


@bentoml.service()
class AudioSpeedUp:
    @bentoml.api()
    def speed_up_audio(
        self,
        context: bentoml.Context,
        audio: t.Annotated[Path, bentoml.validators.ContentType("audio/mpeg")],
        velocity: float,
    ) -> t.Annotated[Path, bentoml.validators.ContentType("audio/mp3")]:
        import os

        from pydub import AudioSegment

        output_path = os.path.join(context.temp_dir, "output.mp3")
        sound = AudioSegment.from_file(audio)
        sound = sound.speedup(velocity)
        sound.export(output_path, format="mp3")
        return Path(output_path)


@bentoml.service()
class TransposeTensor:
    @bentoml.api()
    def transpose(
        self,
        tensor: t.Annotated[torch.Tensor, DType("float32")] = Field(
            description="A 2x4 tensor with float32 dtype"
        ),
    ) -> np.ndarray:
        return torch.transpose(tensor, 0, 1).numpy()


@bentoml.service()
class CountRowsDF:
    @bentoml.api()
    def count_rows(
        self,
        input: t.Annotated[
            pd.DataFrame,
            DataframeSchema(orient="records", columns=["dummy1", "dummy2"]),
        ],
    ) -> int:
        return len(input)
