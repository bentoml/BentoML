from __future__ import annotations

if __name__ == "__main__":
    import argparse

    import bentoml

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None)

    args = parser.parse_args()

    tag = "triton-integration-tensorflow"
    if args.tag:
        tag = f"triton-integration-tensorflow:{args.tag}"

    backend = "docker"
    try:
        builder = bentoml.container.get_backend("buildx")
        assert builder.health()
        backend = "buildx"
    except ValueError:
        print("Buildx not found, using default Docker builder.")
        try:
            builder = bentoml.container.get_backend(backend)
            assert builder.health()
        except ValueError:
            print("Make sure to have Docker running.")
            raise
    finally:
        bentoml.container.build(
            tag, backend=backend, features=["all"], platform="linux/amd64"
        )
