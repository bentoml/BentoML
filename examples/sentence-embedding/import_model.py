import huggingface_hub

import bentoml
from bentoml.models import ModelContext

api = huggingface_hub.HfApi()
repo_info = api.repo_info("sentence-transformers/all-MiniLM-L6-v2")
print(repo_info.sha)

# Save model to BentoML local model store
with bentoml.models.create(
    f"all-MiniLM-L6-v2:{repo_info.sha}",
    module="bentoml.transformers",
    context=ModelContext(framework_name="", framework_versions={}),
    signatures={},
) as model_ref:
    huggingface_hub.snapshot_download(
        "sentence-transformers/all-MiniLM-L6-v2",
        local_dir=model_ref.path_of("/"),
        local_dir_use_symlinks=False,
    )
print(f"Model saved: {model_ref}")
