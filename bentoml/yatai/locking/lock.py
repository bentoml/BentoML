from contextlib import contextmanager

from bentoml.exceptions import LockUnavailable

import logging
import random
import time

from bentoml.yatai.db.stores.lock import LockStore, LOCK_STATUS

logger = logging.getLogger(__name__)


class LockType:
    READ = LOCK_STATUS.read_lock
    WRITE = LOCK_STATUS.write_lock


@contextmanager
def lock(
    db,
    lock_identifier,
    lock_type=LockType.READ,
    timeout_seconds=10,
    timeout_jitter_seconds=1,
    max_retry_count=5,
):
    for i in range(max_retry_count):
        try:
            with db.create_session() as sess:
                lock_obj = LockStore.acquire(sess, lock_type, lock_identifier, 3)
                yield sess, lock_obj
                lock_obj.release(sess)
                return
        except LockUnavailable as e:
            sleep_seconds = timeout_seconds + random.random() * timeout_jitter_seconds
            time.sleep(sleep_seconds)
            logger.warning(
                f"Failed to acquire lock: {e}."
                f"Retrying {max_retry_count - i - 1} more times..."
            )
    raise LockUnavailable(f"Failed to acquire lock after {max_retry_count} attempts")
