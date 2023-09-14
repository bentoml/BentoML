import logging

import transformers

import bentoml

logging.basicConfig(level=logging.WARN)


if __name__ == "__main__":
    # Create Transformers pipelines from pretrained models
    generation_pipeline_1 = transformers.pipeline(
        task="text-generation",
        model="gpt2",
    )
    generation_pipeline_2 = transformers.pipeline(
        task="text-generation",
        model="distilgpt2",
    )
    generation_pipeline_2 = transformers.pipeline(
        task="text-generation",
        model="gpt2-medium",
    )

    classification_pipeline = transformers.pipeline(
        task="text-classification",
        model="bert-base-uncased",
        tokenizer="bert-base-uncased",
    )

    # Save models to BentoML local model store
    m0 = bentoml.transformers.save_model("gpt2-generation", generation_pipeline_1)
    m1 = bentoml.transformers.save_model("distilgpt2-generation", generation_pipeline_2)
    m2 = bentoml.transformers.save_model(
        "gpt2-medium-generation", generation_pipeline_2
    )
    m3 = bentoml.transformers.save_model(
        "bert-base-uncased-classification", classification_pipeline
    )

    print(f"Model saved: {m0}, {m1}, {m2}, {m3}")
