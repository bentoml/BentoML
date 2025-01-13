================
Stream responses
================

Streaming allows you to efficiently handle large or incremental data by sending it in chunks to the client, thus providing better real-time user experiences. BentoML supports streaming responses for various applications, such as large language model (LLM) output and audio synthesis.

LLM output
----------

In BentoML, you can stream LLM output using Python generators. Here's an example using OpenAI's API:

.. code-block:: python
    :caption: `service.py`

    import bentoml
    from typing import Literal, Generator
    from pydantic import BaseModel

    # Define message structure for LLM input
    class Message(BaseModel):
        content: str
        role: Literal['assistant', 'user', 'system']

    @bentoml.service
    class LLMExample:
        def __init__(self) -> None:
            # Initialize your model configuration
            # A dummy example here
            self.model_id = MODEL_ID

        @bentoml.api
        async def generate(self, prompt: str) -> Generator[str, None, None]:
            # Yields text chunks from the LLM response
            from openai import AsyncOpenAI

            # Initialize OpenAI client
            client = AsyncOpenAI()
            message = Message(role="user", content=prompt)

            # Call OpenAI's chat completion API with streaming enabled
            completion = await client.chat.completions.create(
                model=self.model_id,
                messages=[message.model_dump()], # type: ignore
                stream=True,
            )

            # Stream and yield the response chunks
            async for chunk in completion:
                yield chunk.choices[0].delta.content or ""

For more practical examples, see `how you can serve different LLMs with BentoML and vLLM <https://github.com/bentoml/BentoVLLM>`_.

Audio bytes
-----------

Audio streaming is essential for applications such as text-to-speech (TTS), real-time voice assistants, and real-time audio processing. These use cases often require building a WebSocket server to stream audio data to clients.

Here is an example of :doc:`configuring a WebSocket server </build-with-bentoml/websocket>` for streaming audio bytes in BentoML.

.. code-block:: python
    :caption: `service.py`

    import bentoml
    from fastapi import FastAPI, WebSocket
    from typing import Generator

    # Create a FastAPI app
    app = FastAPI()

    @bentoml.service
    @bentoml.asgi_app(app) # Integrate FastAPI app with BentoML
    class TTSExample:
        def __init__(self) -> None:
            # Initialize your TTS engine here
            self.engine = self.setup_tts_engine()

        def setup_tts_engine(self):
            # Configure your TTS engine here
            pass

        def synthesize(self, text: str) -> Generator[bytes, None, None]:
            # Implement your TTS logic here
            pass

        # Define a WebSocket endpoint for streaming audio
        @app.websocket("/ws")
        async def speech(self, websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    # Receive text from client
                    data = await websocket.receive_text()
                    # Stream audio chunks back to client
                    for chunk in self.engine.synthesize(data):
                        await websocket.send_bytes(chunk)
            except Exception as e:
                print(f"Error in WebSocket connection: {e}")
            finally:
                await websocket.close()

Learn more
----------

For more practical examples, see `how to build a voice agent with open-source models <https://github.com/bentoml/BentoVoiceAgent>`_.
