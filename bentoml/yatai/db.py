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

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from bentoml.exceptions import BentoMLException

Base = declarative_base()

logger = logging.getLogger(__name__)


def is_sqlite_db(db_url):
    try:
        return urlparse(db_url).scheme == 'sqlite'
    except ValueError:
        return False


def init_db(db_url):
    from sqlalchemy_utils import database_exists

    extra_db_args = {'echo': True}

    if is_sqlite_db(db_url):
        extra_db_args['connect_args'] = {'check_same_thread': False}
        extra_db_args['echo'] = False
    engine = create_engine(db_url, **extra_db_args)

    if not database_exists(engine.url) and not is_sqlite_db(db_url):
        raise BentoMLException(
            f'Database does not exist or Database name is missing in config '
            f'db.url: {db_url}'
        )
    create_all_or_upgrade_db(engine, db_url)

    return sessionmaker(bind=engine)


@contextmanager
def create_session(session_maker):
    session = session_maker()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise BentoMLException(e)
    finally:
        session.close()


def create_all_or_upgrade_db(engine, db_url):
    # alembic add a lot of import time, so we lazy import
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect

    alembic_config_file = os.path.join(os.path.dirname(__file__), 'alembic.ini')
    alembic_config = Config(alembic_config_file)
    alembic_config.set_main_option('sqlalchemy.url', db_url)

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    if 'deployments' not in tables or 'bentos' not in tables:
        logger.debug('Creating tables')
        Base.metadata.create_all(engine)
        command.stamp(alembic_config, 'head')
    else:
        logger.debug('Upgrading tables to the latest revision')
        command.upgrade(alembic_config, 'heads')
