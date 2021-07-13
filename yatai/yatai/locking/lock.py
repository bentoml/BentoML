from contextlib import contextmanager
from bentoml.exceptions import LockUnavailable

import logging
import random
import time

from bentoml.yatai.db.stores.lock import LockStore, LOCK_STATUS

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_TIMEOUT_JITTER_SECONDS = 1
DEFAULT_MAX_RETRY_COUNT = 5
DEFAULT_TTL_MIN = 3


# enum of read or write lock
class LockType:
    READ = LOCK_STATUS.read_lock
    WRITE = LOCK_STATUS.write_lock


@contextmanager
def lock(
    db,
    locks,
    timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    timeout_jitter_seconds=DEFAULT_TIMEOUT_JITTER_SECONDS,
    max_retry_count=DEFAULT_MAX_RETRY_COUNT,
    ttl_min=DEFAULT_TTL_MIN,
):
    """
    Context manager to acquire operation-level locks on all
    resources defined in the locks parameter
    :param db: instance of bentoml.yatai.db.DB
    :param locks: [
        (lock_identifier: string,
        lock_type: bentoml.yatai.locking.lock.LockType)
    ]
    :param timeout_seconds: amount of time to wait between lock acquisitions
    :param timeout_jitter_seconds: amount of random jitter to add to timeout
    :param max_retry_count: times to retry lock acquisition before failing
    :param ttl_min: amount of time before lock expires
    :return: (sess, lock_objs)
    :exception: LockUnavailable when lock cannot be acquired


    Example Usage:
    ```
    with lock(
        db, [(deployment_id, LockType.WRITE), (bento_id, LockType.READ)]
    ) as (sess, lock_objs):
        # begin critical section
    ```
    """
    if len(locks) < 1:
        raise ValueError("At least one lock needs to be acquired")

    # try to acquire lock
    for i in range(max_retry_count):
        try:
            # create session
            with db.create_session() as sess:
                lock_objs = []
                # acquire all locks in lock list
                for (lock_identifier, lock_type) in locks:
                    lock_obj = LockStore.acquire(
                        sess, lock_type, lock_identifier, ttl_min
                    )
                    lock_objs.append(lock_obj)

                # try to commit all locks to db
                sess.commit()
                logger.debug("Session acquired")
                for lck in locks:
                    op = 'READ' if lck[1] == LOCK_STATUS.read_lock else 'WRITE'
                    logger.debug(f"\t{op} on {lck[0]}")
                start = time.time()

                # return locked session to user
                yield sess, lock_objs

                # release all locks
                for lock_obj in lock_objs:
                    lock_obj.release(sess)
                sess.commit()

                end = time.time()
                logger.debug(f"Session released after {end - start}s")
                return
        except LockUnavailable as e:
            # wait before retrying
            sleep_seconds = timeout_seconds + random.random() * timeout_jitter_seconds
            time.sleep(sleep_seconds)
            logger.warning(
                f"Failed to acquire lock: {e}. "
                f"Retrying {max_retry_count - i - 1} more times..."
            )
    raise LockUnavailable(f"Failed to acquire lock after {max_retry_count} attempts")
