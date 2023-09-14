import asyncio

import bentoml
from bentoml.io import JSON
from bentoml.io import Text

gpt2_generator = bentoml.transformers.get("gpt2-generation:latest").to_runner()
distilgpt2_generator = bentoml.transformers.get(
    "distilgpt2-generation:latest"
).to_runner()
distilbegpt2_medium_generator = bentoml.transformers.get(
    "gpt2-medium-generation:latest"
).to_runner()
bert_base_uncased_classifier = bentoml.transformers.get(
    "bert-base-uncased-classification:latest"
).to_runner()

svc = bentoml.Service(
    "inference_graph",
    runners=[
        gpt2_generator,
        distilgpt2_generator,
        distilbegpt2_medium_generator,
        bert_base_uncased_classifier,
    ],
)


MAX_LENGTH = 128
NUM_RETURN_SEQUENCE = 1


@svc.api(input=Text(), output=JSON())
async def classify_generated_texts(original_sentence: str) -> dict:
    generated_sentences = [
        result[0]["generated_text"]
        for result in await asyncio.gather(
            gpt2_generator.async_run(
                original_sentence,
                max_length=MAX_LENGTH,
                num_return_sequences=NUM_RETURN_SEQUENCE,
            ),
            distilgpt2_generator.async_run(
                original_sentence,
                max_length=MAX_LENGTH,
                num_return_sequences=NUM_RETURN_SEQUENCE,
            ),
            distilbegpt2_medium_generator.async_run(
                original_sentence,
                max_length=MAX_LENGTH,
                num_return_sequences=NUM_RETURN_SEQUENCE,
            ),
        )
    ]

    results = []
    for sentence in generated_sentences:
        score = (await bert_base_uncased_classifier.async_run(sentence))[0]["score"]
        results.append(
            {
                "generated": sentence,
                "score": score,
            }
        )

    return results
