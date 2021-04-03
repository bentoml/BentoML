from contextlib import contextmanager

from bentoml.exceptions import LockUnavailable, BentoMLException

import logging
import random
import time

from bentoml.yatai.db.stores.lock import LockStore, LOCK_STATUS

logger = logging.getLogger(__name__)

class LockType:
    READ = LOCK_STATUS.read_lock
    WRITE = LOCK_STATUS.write_lock


@contextmanager
def lock(db, lock_identifier, type=LockType.READ, timeout_seconds=10, timeout_jitter_seconds=1, max_retry_count=5):
    for i in range(max_retry_count):
        try:
            with db.create_session() as sess:
                LockStore.acquire(sess, type, lock_identifier)
                yield sess
                break
        except LockUnavailable as e:
            sleep_seconds = timeout_seconds + random.random() * timeout_jitter_seconds
            time.sleep(sleep_seconds)
            logger.warning(f"Failed to acquire lock: {e}. Retrying {max_retry_count - i - 1} more times...")
        raise LockUnavailable(f"Failed to acquire lock after {max_retry_count} attempts")