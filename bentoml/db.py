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

from bentoml.exceptions import BentoMLException

Base = declarative_base()


def init_db(db_url):
    # Use default config if not provided
    engine = create_engine(
        db_url, echo=False, connect_args={'check_same_thread': False}
    )
    Base.metadata.create_all(engine)

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
