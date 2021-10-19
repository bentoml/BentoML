import bentoml.spacy

tag, path = bentoml.spacy.projects(
    "clone",
    name="integrations/huggingface_hub",
    repo_or_store="https://github.com/aarnphm/projects",
    branch="v3",
)
