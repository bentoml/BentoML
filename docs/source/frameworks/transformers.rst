========================
Huggingface Transformers
========================

Here's a simple example of serving Huggingface Transformer models with BentoML:

.. code:: python

    import bentoml
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    tag = bentoml.transformers.save_model("text-generation-pipeline", generator)

    # load the model back:
    loaded = bentoml.transformers.load_model("text-generation-pipeline:latest")

    # Load a given model under `Runner` abstraction with `load_runner`
    runner = bentoml.transformers.get(tag).to_runner()
    runner.init_local()
    batched_sentence = [
        "I love you and I want to spend my whole life with you",
        "I hate you, Lyon, you broke my heart.",
    ]
    runner.run(batched_sentence)


.. currentmodule:: bentoml.transformers

.. autofunction:: bentoml.transformers.save_model

.. autofunction:: bentoml.transformers.load_model

.. autofunction:: bentoml.transformers.get

