import time

import pytest

from bentoml.exceptions import LockUnavailable
from bentoml.yatai.db import DB
from bentoml.yatai.locking.lock import lock, LockType
from tests.threading_util import ThreadWithResult, run_delayed_thread


@pytest.fixture()
def db():
    db_url = 'sqlite:///bentoml/storage.db'
    return DB(db_url)


def try_lock(db, id, ttl, timeout):
    try:
        with lock(
            db,
            [(id, LockType.WRITE)],
            ttl_min=ttl,
            timeout_seconds=1,
        ) as (sess, lock_obj):
            time.sleep(timeout)
            return lock_obj
    except LockUnavailable:
        return False


def test_lock_acquisition_on_locked_resource(db):
    # lock held for 10 seconds and expires in 3 min
    lock_1 = ThreadWithResult(target=try_lock, args=(db, 'some_resource', 3, 10))
    lock_2 = ThreadWithResult(target=try_lock, args=(db, 'some_resource', 3, 10))

    # try to acquire lock_2 1 second after lock_1
    run_delayed_thread(lock_1, lock_2)

    # should acquire first lock but fail to acquire second lock
    assert lock_1.result
    assert not lock_2.result


def test_lock_expiry(db):
    # lock held for 1 second and expires in 3 seconds
    lock_1 = ThreadWithResult(target=try_lock, args=(db, 'some_resource', 1/20, 1))

    # lock held for 10 seconds and expires in 3 min
    lock_2 = ThreadWithResult(target=try_lock, args=(db, 'some_resource', 3, 10))

    # try to acquire lock_2 5 seconds after lock_1
    run_delayed_thread(lock_1, lock_2, 5)

    # both should be acquired successfully
    assert lock_1.result
    assert lock_2.result