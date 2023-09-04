=========
Streaming
=========

Starting from BentoML v1.1.4, streaming support has been introduced for Services and Runners, enhancing the ability to handle multiple responses per request.
Streaming responses are especially useful when the data generation process is time-consuming or when you want to start processing the response before the entire sequence is generated.
This implementation is important for real-time data processing, with token streaming for large language models (LLMs) being a prominent application.

.. warning::

   The streaming feature is currently experimental, and only ``bentoml.io.Text`` supports streaming outputs. Always validate your implementation in a
   testing environment before deploying it to production.

Prerequisites
-------------

- BentoML v1.1.4 or later

Generator endpoints
-------------------

You can create streaming endpoints by converting both ``Runnable.method`` and ``service.api`` to generators. The following is
a token streaming example with a large language model (LLM) `facebook/opt-2.7b <https://huggingface.co/facebook/opt-2.7b>`_.

First, save both the model and the tokenizer to the BentoML Model Store by running the following script.

.. code-block:: python
   :caption: `download_model.py`

    import bentoml
    import transformers

    model = "facebook/opt-2.7b"

    bentoml.transformers.save_model('opt-tokenizer', transformers.AutoTokenizer.from_pretrained(model))
    bentoml.transformers.save_model('opt-model', transformers.AutoModelForCausalLM.from_pretrained(model, device_map='cpu'))

.. note::

   The tokenizer is important for preparing the inputs for the model. It breaks down the input text into smaller units, known as tokens, which can represent words, subwords, or even characters, based on its design.
   The primary role of the tokenizer is to translate human-readable text into a format suitable for the model and, subsequently, to decode the model's output back into human-readable text.

Next, create a BentoML Service.

.. code-block:: python
   :caption: `service.py`

    import bentoml
    import torch
    import typing as t

    max_new_tokens = 50
    stream_interval = 2
    context_length = 2048

    class StreamRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            self.tokenizer = bentoml.transformers.load_model("opt-tokenizer")
            self.model = bentoml.transformers.load_model("opt-model")

        @bentoml.Runnable.method()
        async def generate(self, prompt: str) -> t.AsyncGenerator[str, None]:
            input_ids = self.tokenizer(prompt).input_ids

            max_src_len = context_length - max_new_tokens - 1
            input_ids = input_ids[-max_src_len:]
            output_ids = list(input_ids)

            past_key_values = out = token = None

            for i in range(max_new_tokens):
                if i == 0:  # prefill
                    out = self.model(torch.as_tensor([input_ids]), use_cache=True)
                    logits = out.logits
                    past_key_values = out.past_key_values
                else:  # decoding
                    out = self.model(input_ids=torch.as_tensor([[token]]), use_cache=True, past_key_values=past_key_values)
                    logits = out.logits
                    past_key_values = out.past_key_values

                last_token_logits = logits[0, -1, :]

                probs = torch.softmax(last_token_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))
                output_ids.append(token)

                yield self.tokenizer.decode(
                    token,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True
                )

    stream_runner = bentoml.Runner(StreamRunnable)
    svc = bentoml.Service("stream-service", runners=[stream_runner])

    @svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
    async def generate(prompt:str) -> t.AsyncGenerator[str, None]:
        ret = stream_runner.generate.async_stream(prompt)
        return ret

Start the server.

.. code-block:: bash

   $ bentoml serve service:svc

   2023-09-04T11:35:03+0800 [WARNING] [cli] Using lowercased runnable class name 'streamrunnable' for runner.
   2023-09-04T11:35:03+0800 [INFO] [cli] Environ for worker 0: set CPU thread count to 12
   2023-09-04T11:35:03+0800 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "service:svc" can be accessed at http://localhost:3000/metrics.
   2023-09-04T11:35:03+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:svc" listening on http://0.0.0.0:3000 (Press CTRL+C to quit)

This Service streams back the generated tokens one by one, and the response is a sequence of tokens produced by the model for the given input. Specifically, here is what happens
after you send a request to the server.

1. The ``generate`` endpoint is first triggered with the provided prompt.
2. The prompt is tokenized using ``self.tokenizer(prompt).input_ids``.
3. The input is then preprocessed to ensure it is within a specific length (``max_src_len``).
4. The model starts generating tokens based on the provided prompt and the logic described.
5. For each token generated, the token is decoded back to its string representation.
6. The decoded token (as a string) is then streamed back to the client using ``yield``.

Run the following command to test the code:

.. code-block:: bash

   curl -N -X 'POST' 'http://localhost:3000/generate' -d 'What is the meaning of life?'

This returns a stream of text, token by token, produced by the model based on your input prompt. As this example uses an asynchronous generator to stream the response,
you receive parts of the generated text incrementally until the entire sequence (up to ``max_new_tokens``) is sent back. Depending on the model and the tokenizer
used, these tokens might correspond to whole words, parts of words, or even single characters.

Typically, an asynchronous generator is the preferred choice for streaming data, while synchronous generators should still work out of the box.
The syntax of an asynchronous generator looks like the following:

.. code-block:: python

    async def my_gen():
        ...
        yield result

.. note::

   The example above is a simplified version of generating tokens with LLMs. In production, we recommend using `OpenLLM <https://github.com/bentoml/OpenLLM>`_.

Server Sent Events (SSE)
------------------------

Server-Sent Events (SSE) allow servers to push real-time updates to web clients. Once a client establishes an SSE connection, the server can continuously transmit
data without awaiting new requests from the client. This feature is commonly used for transmitting continuous data streams or message updates to browsers.

With BentoML's streaming support, you can easily enable `Server-Sent Events (SSE) <https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events>`_ in your application. To do so:

- Use ``content_type='text/event-stream'`` for the output descriptor, namely ``bentoml.io.Text(content_type='text/event-stream')``.
- Ensure that the returned text data follows the `SSE format <https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format>`_.

Using the example above, here’s how the code might look like to enable SSE:

.. code-block:: python

    import bentoml
    import torch
    import typing as t

    max_new_tokens = 50
    stream_interval = 2
    context_length = 2048

    class StreamRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            self.tokenizer = bentoml.transformers.load_model("opt-tokenizer")
            self.model = bentoml.transformers.load_model("opt-model")

        @bentoml.Runnable.method()
        async def generate(self, prompt: str) -> t.AsyncGenerator[str, None]:
            input_ids = self.tokenizer(prompt).input_ids
            max_src_len = context_length - max_new_tokens - 1
            input_ids = input_ids[-max_src_len:]
            output_ids = list(input_ids)

            past_key_values = out = token = None

            for i in range(max_new_tokens):
                if i == 0:  # prefill
                    out = self.model(torch.as_tensor([input_ids]), use_cache=True)
                    logits = out.logits
                    past_key_values = out.past_key_values
                else:  # decoding
                    out = self.model(input_ids=torch.as_tensor([[token]]), use_cache=True, past_key_values=past_key_values)
                    logits = out.logits
                    past_key_values = out.past_key_values

                last_token_logits = logits[0, -1, :]

                probs = torch.softmax(last_token_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))
                output_ids.append(token)

                decoded_token = self.tokenizer.decode(
                    token,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True
                )

                # Format and yield the token for SSE
                yield f"event: message\ndata: {decoded_token}\n\n"

            # Indicate the end of the stream
            yield "event: end\n\n"

    stream_runner = bentoml.Runner(StreamRunnable)
    svc = bentoml.Service("stream-service", runners=[stream_runner])

    @svc.api(input=bentoml.io.Text(), output=bentoml.io.Text(content_type='text/event-stream'))
    async def generate(prompt: str) -> t.AsyncGenerator[str, None]:
        async for token in stream_runner.generate.async_stream(prompt):
            yield token

In the SSE-enabled code, each decoded token is formatted for SSE using the format ``event: message\ndata: {decoded_token}\n\n``. After all the tokens are sent, a final event ``end``
is sent to signify the end of the stream. This helps clients identify when the data stream is complete.

By integrating SSE with BentoML's streaming capabilities, you can efficiently push real-time updates to the clients, enhancing user experience and application responsiveness.
