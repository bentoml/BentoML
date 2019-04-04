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

    if is_s3_url(model_path):
        temp_dir = tempfile.mkdtemp()
        download_from_s3(model_path, temp_dir)
        model_path = temp_dir
    else:
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)

    model_service = load(model_path)
    name = "bento_rest_api_server"

    server = BentoModelApiServer(name, model_service, port)
    server.start()


if __name__ == '__main__':
    cli()
