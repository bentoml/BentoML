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
from bentoml.server.gunicorn_server import GunicornApplication, get_gunicorn_worker_count
from bentoml.cli.click_utils import DefaultCommandGroup, conditional_argument
from bentoml.deployment.serverless import generate_serverless_bundle, deploy_with_serverless
from bentoml.utils.exceptions import BentoMLException

SERVERLESS_PLATFORMS = ['aws-lambda', 'aws-lambda-py2', 'gcp-function']


def create_bentoml_cli(installed_archive_path=None):
    # pylint: disable=unused-variable

    @click.group(cls=DefaultCommandGroup)
    @click.version_option()
    def bentoml_cli():
        """
        BentoML CLI tool
        """

    # Example Usage: bentoml API_NAME /SAVED_ARCHIVE_PATH --input=INPUT
    @bentoml_cli.command(default_command=True,
                         default_command_usage="API_NAME BENTO_ARCHIVE_PATH --input=INPUT",
                         default_command_display_name="<API_NAME>",
                         help="Run a API defined in saved BentoArchive with cli args as input",
                         context_settings=dict(
                             ignore_unknown_options=True,
                             allow_extra_args=True,
                         ))
    @click.argument('api-name', type=click.STRING)
    @conditional_argument(installed_archive_path is None, 'archive-path', type=click.STRING)
    @click.pass_context
    def run(ctx, api_name, archive_path=installed_archive_path):
        """
        Run an API definied in the BentoService loaded from archive
        """
        model_service = load(archive_path)

        try:
            api = next((api for api in model_service.get_service_apis() if api.name == api_name))
        except StopIteration:
            raise ValueError("Can't find API '{}' in Service '{}'".format(
                api_name, model_service.name))

        api.handle_cli(ctx.args)

    # Example Usage: bentoml info /SAVED_ARCHIVE_PATH
    @bentoml_cli.command()
    @conditional_argument(installed_archive_path is None, 'archive-path', type=click.STRING)
    def info(archive_path=installed_archive_path):
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
    @bentoml_cli.command()
    @conditional_argument(installed_archive_path is None, 'archive-path', type=click.STRING)
    @click.option('--port', type=click.INT, default=BentoAPIServer._DEFAULT_PORT)
    def serve(port, archive_path=installed_archive_path):
        """
        Start REST API server hosting BentoService loaded from archive
        """
        model_service = load(archive_path)
        server = BentoAPIServer(model_service, port=port)
        server.start()

    # Example Usage: bentoml serve-gunicorn ./SAVED_ARCHIVE_PATH --port=PORT --workers=WORKERS
    @bentoml_cli.command()
    @conditional_argument(installed_archive_path is None, 'archive-path', type=click.STRING)
    @click.option('-p', '--port', type=click.INT, default=BentoAPIServer._DEFAULT_PORT)
    @click.option('-w', '--workers', type=click.INT, default=get_gunicorn_worker_count())
    def serve_gunicorn(port, workers, archive_path=installed_archive_path):
        """
        Start REST API gunicorn server hosting BentoService loaded from archive
        """
        model_service = load(archive_path)
        server = BentoAPIServer(model_service, port=port)
        gunicorn_app = GunicornApplication(server.app, port, workers)
        gunicorn_app.run()

    # pylint: enable=unused-variable
    return bentoml_cli


def cli():
    _cli = create_bentoml_cli()

    # Commands created here are mean to be used from generated service archive.  They
    # are used as part of BentoML cli commands only.

    # pylint: disable=unused-variable

    @_cli.command(help='Deploy BentoML archive to AWS Lambda or Google Cloud Function as ' +
                  'REST endpoint with Serverless Framework')
    @click.argument('archive-path', type=click.STRING)
    @click.option('--platform', type=click.Choice([
        'aws-lambda', 'aws-lambda-py2', 'gcp-function', 'aws-sagemaker', 'azure-ml', 'algorithmia'
    ]), required=True)
    @click.option('--region', type=click.STRING)
    @click.option('--stage', type=click.STRING)
    def deploy(archive_path, platform, region, stage):
        if platform in SERVERLESS_PLATFORMS:
            output_path = deploy_with_serverless(platform, archive_path, {
                "region": region,
                "stage": stage
            })
            click.echo('BentoML: ', nl=False)
            click.secho('Deploy to {platform} complete!'.format(platform=platform), fg='green')
            click.secho(
                'Deployment archive is saved at {output_path}'.format(output_path=output_path),
                fg='green')
            return
        else:
            raise BentoMLException(
                'Deploying with "--platform={platform}" is not supported in the current version of BentoML'
                .format(platform=platform))

    # pylint: enable=unused-variable

    _cli()


if __name__ == '__main__':
    cli()
