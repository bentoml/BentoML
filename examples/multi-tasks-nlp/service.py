from __future__ import annotations

import typing as t
import asyncio

import pydantic
from warmup import TEXT
from warmup import CATEGORIES
from warmup import MAX_LENGTH
from warmup import CATEGORICAL_THRESHOLD

import bentoml

summarizer = bentoml.transformers.get("summarizer-pipeline").to_runner()
categorizer = bentoml.transformers.get("categorizer-pipeline").to_runner()

svc = bentoml.Service(name="multi-tasks-nlp", runners=[summarizer, categorizer])


@svc.api(input=bentoml.io.Text.from_sample(TEXT), output=bentoml.io.Text())
async def summarize(text: str) -> str:
    generated = await summarizer.async_run(text, max_length=MAX_LENGTH)
    return generated[0]["summary_text"]


@svc.api(
    input=bentoml.io.JSON.from_sample({"text": TEXT, "categories": CATEGORIES}),
    output=bentoml.io.JSON(),
)
async def categorize(
    input_data: dict[t.Literal["text", "categories"], str | list[str]]
) -> dict[str, float]:
    predictions = await categorizer.async_run(
        input_data["text"], input_data["categories"], multi_label=True
    )
    return {
        c: p
        for c, p in zip(predictions["labels"], predictions["scores"])
        if p > CATEGORICAL_THRESHOLD
    }


class GeneralAnalysisOutput(pydantic.BaseModel):
    summary: str
    categories: dict[str, float]


@svc.api(
    input=bentoml.io.JSON.from_sample({"text": TEXT, "categories": CATEGORIES}),
    output=bentoml.io.JSON.from_sample(
        GeneralAnalysisOutput(
            summary=" Hunter Schafer wore a bias-cut white silk skirt, a single ivory-colored feather and nothing else . The look debuted earlier this month at fashion house Ann Demeulemeester's show in Paris . It was designed by Ludovic de Saint Sernin, the label's creative director since December .",
            categories={
                "entertainment": 0.5805651545524597,
                "world": 0.5592136979103088,
            },
        )
    ),
)
async def make_analysis(
    input_data: dict[t.Literal["text", "categories"], str | list[str]]
) -> GeneralAnalysisOutput:
    text, categories = input_data["text"], input_data["categories"]
    results = [
        res
        for res in await asyncio.gather(
            summarizer.async_run(text, max_length=MAX_LENGTH),
            categorizer.async_run(text, categories, multi_label=True),
        )
    ]
    return GeneralAnalysisOutput(
        summary=results[0][0]["summary_text"],
        categories={
            c: p
            for c, p in zip(results[1]["labels"], results[1]["scores"])
            if p > CATEGORICAL_THRESHOLD
        },
    )
