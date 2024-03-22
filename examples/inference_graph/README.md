# BentoML Inference Graph Tutorial

This is a sample project demonstrating model inference graph of [BentoML](https://github.com/bentoml)
with Huggingface Transformers.

In this project, we will use pretrained text generation models and a pretrained text classification model. We will then build a service that accepts a text input, passes the input to the two text generation models,
classify each generated paragraph with the classification model, and return both generated paragraphs with their classification
scores. The service will be served via HTTP and containerized as a docker image for production deployment.

### Install Dependencies

Install python packages required for running this project:
```bash
python -m venv inference_graph
source venv/bin/activate
pip install -r ./requirements.txt
```

### Serving the model

The service definition below achieves the inference graph logic described above.

First, we create two classes using different GPT text generation models (GPT2 and DistilGPT2 respectively. Although
they are no longer prevelant models, they are comparatively small in size, which aligns with the goal of demonstrating
BentoML inference graph functionality with minimal resources.) Thier corresponding service api is wrapped by `@bentoml.api()` decorator, which takes a sentence as a imput, passed to thier transormer pipeline, finally output text auto completion.

Second, following the similar pattern, we create a classifier class called `BertBaseUncased`. Its service api will take the text auto compeletion result by the above services one by one, and generate a score for each text compeletion to classify
the results. (For example, if one text completion says "This movie is bad" and the other completion says "This movie is good", the classification score will differ.)

Finally, we will have a InferenceGraph class to use all of the above procedures. its service api `generate_score` would be the service endpoint to users. Notice that at the beginning of the class, we instancialize the GPTs and the Classifier. This enables BentoML to call these services in a graph manner.

Inside the api, we use `asyncio.gather` to gather the text autocompletions, store them into `generated_sentences`. We use the `to_async` function so that the two GPT services will run in parallel (It is equivalent to defining the GPT apis as `async def`; it is just the BentoML's way of defining asynchronous functions, so that each api can be called both synchrnously and asynchrounously). Then, we pass the `generated_sentences` one by one to the classifier service, and return the classify score as final result.

```python
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

@bentoml.service()
class BertBaseUncased:
    def __init__(self):
        self.classification_pipeline = transformers.pipeline(
            task="text-classification",
            model="bert-base-uncased",
            tokenizer="bert-base-uncased",
        )

    @bentoml.api()
    async def classify_generated_texts(self, sentence: str) -> float | str:
        score = self.classification_pipeline(sentence)[0]["score"] # type: ignore
        return score

@bentoml.service()
class InferenceGraph:
    gpt2_generator = bentoml.depends(GPT2)
    distilgpt2_generator = bentoml.depends(DistilGPT2)
    bert_classifier = bentoml.depends(BertBaseUncased)

    @bentoml.api()
    async def generate_score(self, original_sentence: str = "I have an idea!") -> t.List[t.Dict[str, t.Any]]:
        generated_sentences = [ # type: ignore
            result[0]["generated_text"]
            for result in await asyncio.gather( # type: ignore
                self.gpt2_generator.to_async.generate( # type: ignore
                    original_sentence,
                    max_length=MAX_LENGTH,
                    num_return_sequences=NUM_RETURN_SEQUENCE,
                ),
                self.distilgpt2_generator.to_async.generate( # type: ignore
                    original_sentence,
                    max_length=MAX_LENGTH,
                    num_return_sequences=NUM_RETURN_SEQUENCE,
                ),
            )
        ]

        results = []
        for sentence in generated_sentences: # type: ignore
            score = await self.bert_classifier.to_async.classify_generated_texts(sentence) # type: ignore
            results.append(
                {
                    "generated": sentence,
                    "score": score,
                }
            )

        return results
```

To serve the model locally, run `bentoml serve .`

```bash
bentoml serve .
2024-03-22T19:25:24+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:InferenceGraph" listening on http://localhost:3000 (Press CTRL+C to quit)
```

Open your web browser at http://0.0.0.0:3000 to view the Bento UI for sending test requests.

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
