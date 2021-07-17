#  Copyright (c) 2021 Atalaya Tech, Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==========================================================================
#

import time

import pytest

from bentoml.exceptions import LockUnavailable
from bentoml.yatai.db import DB
from bentoml.yatai.locking.lock import LockType, lock
from tests._internal.utils.threading import ThreadWithResult, run_delayed_thread


@pytest.fixture(name="db")
def fixture_db():
    db_url = "sqlite:///bentoml/storage.db"
    return DB(db_url)


def try_lock(db, resource_id, ttl, timeout):
    try:
        with lock(
            db, [(resource_id, LockType.WRITE)], ttl_min=ttl, timeout_seconds=1
        ) as (
            _,
            lock_obj,
        ):
            time.sleep(timeout)
            return lock_obj
    except LockUnavailable:
        return False


def test_lock_acquisition_on_locked_resource(db):
    # lock held for 10 seconds and expires in 3 min
    lock_1 = ThreadWithResult(target=try_lock, args=(db, "some_resource", 3, 10))
    lock_2 = ThreadWithResult(target=try_lock, args=(db, "some_resource", 3, 10))

    # try to acquire lock_2 1 second after lock_1
    run_delayed_thread(lock_1, lock_2)

    # should acquire first lock but fail to acquire second lock
    assert lock_1.result
    assert not lock_2.result


def test_lock_expiry(db):
    # lock held for 1 second and expires in 3 seconds
    lock_1 = ThreadWithResult(target=try_lock, args=(db, "some_resource", 1 / 20, 1))

    # lock held for 10 seconds and expires in 3 min
    lock_2 = ThreadWithResult(target=try_lock, args=(db, "some_resource", 3, 10))

    # try to acquire lock_2 5 seconds after lock_1
    run_delayed_thread(lock_1, lock_2, 5)

    # both should be acquired successfully
    assert lock_1.result
    assert lock_2.result
