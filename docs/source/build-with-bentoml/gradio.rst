====================
Add a UI with Gradio
====================

`Gradio <https://github.com/gradio-app/gradio>`_ is an open-source Python library that allows developers to quickly build a web-based user interface (UI) for AI models. BentoML provides a straightforward API to integrate Gradio for serving models with its UI.

Prerequisites
-------------

The integration requires FastAPI and Gradio. Install them using ``pip``.

.. code-block:: bash

    pip install fastapi gradio

Basic usage
-----------

Follow the steps below to integrate Gradio with a BentoML Service.

1. Start by preparing a BentoML Service. Here's an example using a text summarization model from the :doc:`/get-started/hello-world` guide:

   .. code-block:: python

        import bentoml
        import torch
        from transformers import pipeline

        EXAMPLE_INPUT = "Breaking News: In an astonishing turn of events, the small \
        town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, \
        Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' \
        Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped \
        a record-breaking 20 feet into the air to catch a fly. The event, which took \
        place in Thompson's backyard, is now being investigated by scientists for potential \
        breaches in the laws of physics. Local authorities are considering a town festival \
        to celebrate what is being hailed as 'The Leap of the Century."


        @bentoml.service(resources={"cpu": "4"})
        class Summarization:
            def __init__(self) -> None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.pipeline = pipeline("summarization", device=device)

            @bentoml.api(batchable=True)
            def summarize(self, texts: list[str]) -> list[str]:
                results = self.pipeline(texts)
                return [item["summary_text"] for item in results]

2. Define a helper function to create a Gradio UI for. It retrieves the current Service instance via ``get_current_service()`` and invokes its exposed API method.

   .. code-block:: python

      def summarize_text(text: str) -> str:
          svc_instance = bentoml.get_current_service()
          return svc_instance.summarize([text])[0]

3. Set up a Gradio interface by specifying the function to wrap, and defining the input and output components. For more information, see `the Gradio documentation <https://www.gradio.app/docs/gradio/interface>`_.

   .. code-block:: python

        import gradio as gr

        def summarize_text(text: str) -> str:
            svc_instance = bentoml.get_current_service()
            return svc_instance.summarize([text])[0]

        # Configure a Gradio UI
        io = gr.Interface(
            fn=summarize_text, # Wrap the UI around the function defined above
            inputs=[gr.Textbox(lines=5, label="Enter Text", value=EXAMPLE_INPUT)],
            outputs=[gr.Textbox(label="Summary Text")],
            title="Summarization",
            description="Enter text to get summarized text.",
        )

        @bentoml.service(resources={"cpu": "4"})
        class Summarization:
              ...


4. Use the ``@bentoml.gradio.mount_gradio_app`` decorator to mount the Gradio UI (``io``) at a custom path (``/ui``). This makes it accessible as part of the Service's web server:

   .. code-block:: python

        ...

        @bentoml.service(resources={"cpu": "4"})
        @bentoml.gradio.mount_gradio_app(io, path="/ui")
        class Summarization:
              ...

5. Start the Service using ``bentoml serve``, and access the Gradio UI at ``http://localhost:3000/ui``. You can also call BentoML’s API endpoint ``summarize`` at ``http://localhost:3000/``.

   .. code-block:: python

        bentoml serve service:Summarization

   .. image:: ../../_static/img/build-with-bentoml/gradio/gradio-ui-bentoml.png
      :alt: Gradio UI for a BentoML Service

Visit this `example <https://github.com/bentoml/BentoGradio>`_ to view the full demo code.
