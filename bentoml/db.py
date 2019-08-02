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

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from bentoml import config
from bentoml.exceptions import BentoMLException

# sql alchemy config
engine = create_engine(
    config.get('db', 'engine'), echo=False, connect_args={'check_same_thread': False}
)
Base = declarative_base()
Session = sessionmaker(bind=engine)


@contextmanager
def create_session():
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise BentoMLException(message=e)
    finally:
        session.close()


def initialize_db():
    Base.metadata.create_all(engine)
