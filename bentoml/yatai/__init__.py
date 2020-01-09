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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging


from bentoml import config


logger = logging.getLogger(__name__)


def get_yatai_service(
    channel_address=None, db_url=None, repo_base_url=None, default_namespace=None
):
    if channel_address is not None:
        import grpc
        from bentoml.proto.yatai_service_pb2_grpc import YataiStub

        if db_url is not None:
            logger.warning(
                "Config 'db_url' is ignored in favor of remote YataiService at `%s`",
                channel_address,
            )
        if repo_base_url is not None:
            logger.warning(
                "Config 'repo_base_url:%s' is ignored in favor of remote YataiService "
                "at `%s`",
                repo_base_url,
                channel_address,
            )
        if default_namespace is not None:
            logger.warning(
                "Config 'default_namespace:%s' is ignored in favor of remote "
                "YataiService at `%s`",
                default_namespace,
                channel_address,
            )
        logger.debug("Using BentoML with remote Yatai server: %s", channel_address)

        channel = grpc.insecure_channel(channel_address)
        return YataiStub(channel)
    else:
        from bentoml.yatai.yatai_service_impl import YataiService

        logger.debug("Using BentoML with local Yatai server")

        default_namespace = default_namespace or config().get(
            'deployment', 'default_namespace'
        )
        repo_base_url = repo_base_url or config().get('default_repository_base_url')
        db_url = db_url or config().get('db', 'url')

        return YataiService(
            db_url=db_url,
            repo_base_url=repo_base_url,
            default_namespace=default_namespace,
        )


__all__ = ["get_yatai_service"]
