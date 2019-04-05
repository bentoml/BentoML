# BentoML - Machine Learning Toolkit for packaging and deploying models
# Copyright (C) 2019 Atalaya Tech, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import os
import tempfile

from bentoml.loader import load
from bentoml.server import BentoModelApiServer
from bentoml.utils.s3 import is_s3_url, download_from_s3


def load_model_service(model_path):
    if is_s3_url(model_path):
        temp_dir = tempfile.mkdtemp()
        download_from_s3(model_path, temp_dir)
        model_path = temp_dir
    else:
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)

    model_service = load(model_path)
    return model_service


@click.group()
@click.version_option()
def cli():
    pass


# TODO: allow string to be a value for input as well
# bentoml serve --model=/MODEL_PATH --api-name=API_FUNC_NAME --input=/INPUT_PATH
@cli.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option('--model-path', type=click.Path(exists=True), required=True)
@click.option('--api-name', type=click.STRING)
@click.pass_context
def run(ctx, model_path, api_name):
    model_service = load_model_service(model_path)
    service_apis = model_service.get_service_apis()

    matched_api_index = next(
        (index for (index, d) in enumerate(service_apis) if d.name == api_name), None)

    if matched_api_index is None:
        raise ValueError("Can't find api name inside model {}".format(api_name))
    else:
        matched_api = service_apis[matched_api_index]

    matched_api.handler.handle_cli(ctx.args, matched_api.func)


# bentoml serve --model=/MODEL_PATH --port=PORT_NUM
@cli.command()
@click.option('--model-path', type=click.Path(exists=True), required=True)
@click.option('--port', type=click.INT)
def serve(model_path, port):
    """
    serve command for bentoml cli tool.  It will start a rest API server

    """
    port = port if port is not None else 5000

    model_service = load_model_service(model_path)
    name = "bento_rest_api_server"

    server = BentoModelApiServer(name, model_service, port)
    server.start()


if __name__ == '__main__':
    cli()
