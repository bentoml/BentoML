if __name__ == "__main__":
    import bentoml

    bentoml.bentos.build(
        "service:svc",
        include=["/model_repository", "/data/*.png", "service.py"],
        exclude=["/__pycache__", "/venv"],
        docker={"base_image": "nvcr.io/nvidia/tritonserver:22.12-py3"},
    )
