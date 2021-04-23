# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import click
import logging
from dependency_injector.wiring import Provide, inject

from bentoml.configuration.containers import BentoMLContainer


logger = logging.getLogger(__name__)


@inject
def add_yatai_service_sub_command(
    cli,
    default_db_url: str = Provide[BentoMLContainer.yatai_database_url],
    default_repository_type: str = Provide[
        BentoMLContainer.config.yatai.repository.type
    ],
    default_file_system_directory: str = Provide[
        BentoMLContainer.yatai_file_system_directory
    ],
    default_s3_url: str = Provide[BentoMLContainer.config.yatai.repository.s3.url],
    default_gcs_url: str = Provide[BentoMLContainer.config.yatai.repository.gcs.url],
):
    # pylint: disable=unused-variable

    @cli.command(help='Start BentoML YataiService for model management and deployment')
    @click.option(
        '--db-url',
        type=click.STRING,
        help='Database URL following RFC-1738, and usually can include username, '
        'password, hostname, database name as well as optional keyword arguments '
        'for additional configuration',
        envvar='BENTOML_DB_URL',
        default=default_db_url,
    )
    @click.option(
        '--repo-base-url',
        type=click.STRING,
        help='Deprecated option for base URL for storing BentoML saved bundle files, '
        'this can be a file system path(POSIX/Windows), or a S3 URL, '
        'usually starting with "s3://"',
        envvar='BENTOML_REPO_BASE_URL',
    )
    @click.option(
        '--grpc-port',
        type=click.INT,
        default=50051,
        help='Port to run YataiService gRPC server',
        envvar='BENTOML_GRPC_PORT',
    )
    @click.option(
        '--ui-port',
        type=click.INT,
        default=3000,
        help='Port to run YataiService Web UI server',
        envvar='BENTOML_WEB_UI_PORT',
    )
    @click.option(
        '--ui/--no-ui',
        default=True,
        help='Run YataiService with or without Web UI, when running with --no-ui, it '
        'will only run the gRPC server',
        envvar='BENTOML_ENABLE_WEB_UI',
    )
    @click.option(
        '--web-prefix-path',
        type=click.STRING,
        default='.',
        help='Add a location prefix to the URL when running YataiService'
        'behind a reverse proxy server',
        envvar='BENTOML_YATAI_WEB_PREFIX_PATH',
    )
    @click.option(
        '--repository-type',
        type=click.Choice(['file_system', 's3', 'gcs'], case_sensitive=False),
        default=default_repository_type,
        help='Type of the repository implementation',
        envvar='BENTOML_YATAI_REPOSITORY_TYPE',
    )
    @click.option(
        '--file-system-directory',
        type=click.STRING,
        default=default_file_system_directory,
        help='Specifies the directory path for the file system repository type',
        envvar='BENTOML_YATAI_FILE_SYSTEM_DIRECTORY',
    )
    @click.option(
        '--s3-url',
        type=click.STRING,
        default=default_s3_url,
        help='Specifies the S3 URL for the S3 repository type',
        envvar='BENTOML_YATAI_S3_URL',
    )
    @click.option(
        '--gcs-url',
        type=click.STRING,
        default=default_gcs_url,
        help='Specifies the GCS URL for the GCS repository type',
        envvar='BENTOML_YATAI_GCS_URL',
    )
    def yatai_service_start(
        db_url,
        repo_base_url,
        grpc_port,
        ui_port,
        ui,
        web_prefix_path,
        repository_type,
        file_system_directory,
        s3_url,
        gcs_url,
    ):
        from bentoml.utils.s3 import is_s3_url
        from bentoml.utils.gcs import is_gcs_url

        if repo_base_url:
            logger.warning(
                "Option --repo-base-url has been deprecated but is still supported "
                "in the current release. Consider using --repository-type and its "
                "corresponding options in the upcoming releases. "
            )
            if is_s3_url(repo_base_url):
                repository_type = "s3"
                s3_url = repo_base_url
            elif is_gcs_url(repo_base_url):
                repository_type = "gcs"
                gcs_url = repo_base_url
            else:
                repository_type = "file_system"
                file_system_directory = repo_base_url

        from bentoml.yatai.yatai_service import start_yatai_service_grpc_server

        start_yatai_service_grpc_server(
            db_url,
            grpc_port,
            ui_port,
            ui,
            web_prefix_path,
            repository_type,
            file_system_directory,
            s3_url,
            gcs_url,
        )
