import click
import os
from bentoml.loader import load
from bentoml.server import BentoModelApiServer


@click.group()
@click.version_option()
def cli():
    pass


# bentoml serve --model=/MODEL_PATH --port=PORT_NUM
@cli.command()
@click.option('--model-path', type=click.Path(exists=True), required=True)
@click.option('--port', type=click.INT)
def serve(model_path, port):
    """
    serve command for bentoml cli tool.  It will start a rest API server

    """
    port = port if port is not None else 5000

    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)

    model_service = load(model_path)
    name = "bento_rest_api_server"

    server = BentoModelApiServer(name, model_service, port)
    server.start()


if __name__ == '__main__':
    cli()
