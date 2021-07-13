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

import logging
import os
from contextlib import contextmanager

from urllib.parse import urlparse

from bentoml.exceptions import BentoMLException, LockUnavailable
from bentoml.yatai.db.base import Base
from bentoml.yatai.db.stores.deployment import DeploymentStore
from bentoml.yatai.db.stores.label import LabelStore
from bentoml.yatai.db.stores.metadata import MetadataStore

logger = logging.getLogger(__name__)


def is_postgresql_db(db_url):
    try:
        return urlparse(db_url).scheme == 'postgresql'
    except ValueError:
        return False


def is_sqlite_db(db_url):
    try:
        return urlparse(db_url).scheme == 'sqlite'
    except ValueError:
        return False


class DB(object):
    def __init__(self, db_url):
        from sqlalchemy import create_engine
        from sqlalchemy_utils import database_exists
        from sqlalchemy.orm import sessionmaker

        extra_db_args = {'echo': True}

        self.db_url = db_url
        if is_sqlite_db(db_url):
            extra_db_args['connect_args'] = {'check_same_thread': False}
            extra_db_args['echo'] = False
        elif is_postgresql_db(db_url):
            extra_db_args['connect_args'] = {'application_name': 'yatai'}
            extra_db_args['pool_pre_ping'] = True

        self.engine = create_engine(db_url, **extra_db_args)

        if not database_exists(self.engine.url) and not is_sqlite_db(db_url):
            raise BentoMLException(
                f'Database does not exist or Database name is missing in config '
                f'db.url: {db_url}'
            )

        self.create_all_or_upgrade_db()
        self.session_maker = sessionmaker(bind=self.engine)
        self._setup_stores()

    def _setup_stores(self):
        self.deployment_store = DeploymentStore()
        self.metadata_store = MetadataStore()
        self.label_store = LabelStore()

    @contextmanager
    def create_session(self):
        session = self.session_maker()
        try:
            yield session
            session.commit()
        except LockUnavailable as e:
            # rollback if lock cannot be acquired, bubble error up
            session.rollback()
            raise LockUnavailable(e)
        except Exception as e:
            session.rollback()
            raise BentoMLException(e)
        finally:
            session.close()

    def create_all_or_upgrade_db(self):
        # alembic add a lot of import time, so we lazy import
        from alembic import command
        from alembic.config import Config
        from sqlalchemy import inspect

        alembic_config_file = os.path.join(os.path.dirname(__file__), '../alembic.ini')
        alembic_config = Config(alembic_config_file)
        alembic_config.set_main_option('sqlalchemy.url', self.db_url)

        inspector = inspect(self.engine)
        tables = inspector.get_table_names()

        if 'deployments' not in tables or 'bentos' not in tables:
            logger.debug('Creating tables')
            Base.metadata.create_all(self.engine)
            command.stamp(alembic_config, 'head')
        else:
            logger.debug('Upgrading tables to the latest revision')
            command.upgrade(alembic_config, 'heads')
