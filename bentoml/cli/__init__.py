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

    if os.path.isabs(model_path) is False:
        current_path = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(current_path, model_path)

    model_service = load(model_path)
    name = "bento_rest_api_server"

    server = BentoModelApiServer(name, model_service, port)
    server.start()


if __name__ == '__main__':
    cli()
