import asyncio
import typing as t

import transformers

import bentoml
<<<<<<< HEAD
=======

import transformers
import typing as t

>>>>>>> 0392d279 (docs: initialized inference_graph eexample)

MAX_LENGTH = 128
NUM_RETURN_SEQUENCE = 1

<<<<<<< HEAD

=======
>>>>>>> 0392d279 (docs: initialized inference_graph eexample)
@bentoml.service()
class GPT2:
    def __init__(self):
        self.generation_pipeline_1 = transformers.pipeline(
            task="text-generation",
            model="gpt2",
        )

    @bentoml.api()
    def generate(self, sentence: str) -> t.List[t.Any]:
        return self.generation_pipeline_1(sentence)

<<<<<<< HEAD

=======
>>>>>>> 0392d279 (docs: initialized inference_graph eexample)
@bentoml.service()
class DistilGPT2:
    def __init__(self):
        self.generation_pipeline_2 = transformers.pipeline(
            task="text-generation",
            model="distilgpt2",
        )

    @bentoml.api()
    def generate(self, sentence: str) -> t.List[t.Any]:
        return self.generation_pipeline_2(sentence)

<<<<<<< HEAD

=======
>>>>>>> 0392d279 (docs: initialized inference_graph eexample)
@bentoml.service()
class BertBaseUncased:
    def __init__(self):
        self.classification_pipeline = transformers.pipeline(
            task="text-classification",
            model="bert-base-uncased",
            tokenizer="bert-base-uncased",
        )
<<<<<<< HEAD

    @bentoml.api()
    def classify_generated_texts(self, sentence: str) -> float | str:
        score = self.classification_pipeline(sentence)[0]["score"]  # type: ignore
        return score


=======
    
    @bentoml.api()
    def classify_generated_texts(self, sentence: str) -> float | str:
        score = self.classification_pipeline(sentence)[0]["score"] # type: ignore
        return score

>>>>>>> 0392d279 (docs: initialized inference_graph eexample)
@bentoml.service()
class InferenceGraph:
    gpt2_generator = bentoml.depends(GPT2)
    distilgpt2_generator = bentoml.depends(DistilGPT2)
    bert_classifier = bentoml.depends(BertBaseUncased)
<<<<<<< HEAD

    @bentoml.api()
    async def generate_score(
        self, original_sentence: str = "I have an idea!"
    ) -> t.List[t.Dict[str, t.Any]]:
        generated_sentences = [  # type: ignore
            result[0]["generated_text"]
            for result in await asyncio.gather(  # type: ignore
                self.gpt2_generator.to_async.generate(  # type: ignore
=======
    
    @bentoml.api()
    async def generate_score(self, original_sentence: str = "I have an idea!") -> t.List[t.Dict[str, t.Any]]:
        generated_sentences = [ # type: ignore
            result[0]["generated_text"]
            for result in await asyncio.gather( # type: ignore
                self.gpt2_generator.to_async.generate( # type: ignore
>>>>>>> 0392d279 (docs: initialized inference_graph eexample)
                    original_sentence,
                    max_length=MAX_LENGTH,
                    num_return_sequences=NUM_RETURN_SEQUENCE,
                ),
<<<<<<< HEAD
                self.distilgpt2_generator.to_async.generate(  # type: ignore
=======
                self.distilgpt2_generator.to_async.generate( # type: ignore
>>>>>>> 0392d279 (docs: initialized inference_graph eexample)
                    original_sentence,
                    max_length=MAX_LENGTH,
                    num_return_sequences=NUM_RETURN_SEQUENCE,
                ),
            )
        ]

        results = []
<<<<<<< HEAD
        for sentence in generated_sentences:  # type: ignore
            score = await self.bert_classifier.to_async.classify_generated_texts(
                sentence
            )  # type: ignore
=======
        for sentence in generated_sentences: # type: ignore
            score = await self.bert_classifier.to_async.classify_generated_texts(sentence) # type: ignore
>>>>>>> 0392d279 (docs: initialized inference_graph eexample)
            results.append(
                {
                    "generated": sentence,
                    "score": score,
                }
            )

<<<<<<< HEAD
        return results
=======
        return results
>>>>>>> 0392d279 (docs: initialized inference_graph eexample)
