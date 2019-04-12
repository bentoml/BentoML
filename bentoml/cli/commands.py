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

import json

import click

from bentoml.archive import load
from bentoml.server import BentoAPIServer
from bentoml.cli.default_command_group import DefaultCommandGroup


@click.group(cls=DefaultCommandGroup)
@click.version_option()
def cli():
    """
    BentoML CLI tool
    """


# Example Usage: bentoml API_NAME /SAVED_ARCHIVE_PATH --input=INPUT
@cli.command(default_command=True, context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument('api-name', type=click.STRING)
@click.argument('archive-path', type=click.STRING)
@click.pass_context
def run(ctx, api_name, archive_path):
    """
    Run an API definied in the BentoService loaded from archive
    """
    model_service = load(archive_path)

    try:
        api = next((api for api in model_service.get_service_apis() if api.name == api_name))
    except StopIteration:
        raise ValueError("Can't find API '{}' in Service '{}'".format(api_name, model_service.name))

    api.handler.handle_cli(ctx.args, api.func, api.options)


# Example Usage: bentoml info /SAVED_ARCHIVE_PATH
@cli.command()
@click.argument('archive-path', type=click.STRING)
def info(archive_path):
    """
    List all APIs definied in the BentoService loaded from archive
    """
    model_service = load(archive_path)
    service_apis = model_service.get_service_apis()
    output = json.dumps(
        dict(name=model_service.name, version=model_service.version,
             apis=[api.name for api in service_apis]), indent=2)
    print(output)


# Example Usage: bentoml serve ./SAVED_ARCHIVE_PATH --port=PORT
@cli.command()
@click.argument('archive-path', type=click.STRING)
@click.option('--port', type=click.INT, default=BentoAPIServer._DEFAULT_PORT)
def serve(archive_path, port):
    """
    Start REST API server, hosting BentoService loaded from archive
    """
    model_service = load(archive_path)
    server = BentoAPIServer(model_service, port=port)
    server.start()


if __name__ == '__main__':
    cli()
