import logging

import transformers

import bentoml

logging.basicConfig(level=logging.WARN)


if __name__ == "__main__":
    # Create Transformers pipelines from pretrained models
    pipeline1 = transformers.pipeline(
        task="text-classification",
        model="bert-base-uncased",
        tokenizer="bert-base-uncased",
    )
    pipeline2 = transformers.pipeline(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    pipeline3 = transformers.pipeline(
        task="text-classification", model="ProsusAI/finbert"
    )

    # Save models to BentoML local model store
    m1 = bentoml.transformers.save_model("bert-base-uncased", pipeline1)
    m2 = bentoml.transformers.save_model("distilbert", pipeline2)
    m3 = bentoml.transformers.save_model("prosusai-finbert", pipeline3)

    print(f"Model saved: {m1}, {m2}, {m3}")
