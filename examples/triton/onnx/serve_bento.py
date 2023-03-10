from __future__ import annotations

if __name__ == "__main__":
    import time
    import argparse

    import bentoml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grpc", action="store_true", default=False, help="Whether to serve as gRPC."
    )
    parser.add_argument("--tag", type=str, default=None)

    args = parser.parse_args()

    tag = "triton-integration-onnx"
    if args.tag:
        tag = f"triton-integration-onnx:{args.tag}"

    server_type = "grpc" if args.grpc else "http"

    try:
        bento = bentoml.get(tag)
    except bentoml.exceptions.NotFound:
        raise ValueError(
            "Bento is not yet built. Make sure to run 'python3 build_bento.py' and try to run this script again."
        )
    else:
        bento = bentoml.get(tag)
        server = bentoml.serve(
            bento,
            server_type=server_type,
            production=True,
        )
        try:
            while True:
                bentoml.client.Client.wait_until_server_is_ready(
                    server.host, server.port, 1000
                )
                client = bentoml.client.Client.from_url(
                    f"http://localhost:{server.port}"
                )
                print(
                    "ONNX config:",
                    client.model_config(
                        inp={"model_name": "onnx_yolov5s", "protocol": "grpc"}
                    ),
                )
                time.sleep(1e9)
        except KeyboardInterrupt:
            server.stop()
            raise SystemExit(0)
