import bentoml
import transformers

pipe = transformers.pipeline("text-classification")

bentoml.transformers.save_model(
  "text-classification-pipe",
  pipe,
  signatures={
    "__call__": {"batchable": True}  # Enable dynamic batching for model
  }
)
