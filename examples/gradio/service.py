from __future__ import annotations  # I001

import bentoml

with bentoml.importing():
    import torch
    import gradio as gr
    from transformers import pipeline

EXAMPLE_INPUT = "Breaking News: In an astonishing turn of events, the small \
town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, \
Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' \
Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped \
a record-breaking 20 feet into the air to catch a fly. The event, which took \
place in Thompson's backyard, is now being investigated by scientists for potential \
breaches in the laws of physics. Local authorities are considering a town festival \
to celebrate what is being hailed as 'The Leap of the Century."


def summarize_text(text: str) -> str:
    svc_instance = bentoml.get_current_service()
    return svc_instance.summarize([text])[0]


io = gr.Interface(
    fn=summarize_text,
    inputs=[gr.Textbox(lines=5, label="Enter Text", value=EXAMPLE_INPUT)],
    outputs=[gr.Textbox(label="Summary Text")],
    title="Summarization",
    description="Enter text to get summarized text.",
)


@bentoml.service(resources={"cpu": "4"})
@bentoml.gradio.mount_gradio_app(io, path="/ui")
class Summarization:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = pipeline("summarization", device=device)

    @bentoml.api(batchable=True)
    def summarize(self, texts: list[str]) -> list[str]:
        results = self.pipeline(texts)
        return [item["summary_text"] for item in results]
