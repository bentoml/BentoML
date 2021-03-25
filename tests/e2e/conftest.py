import time
import pytest


@pytest.fixture()
def postgres_db_url():
    db_url = 'postgresql://postgres:postgres@localhost/bentoml:5432'
    from sqlalchemy_utils import create_database

    create_database(db_url)
    time.sleep(60)

    yield db_url
