from bentoml.exceptions import LockUnavailable
import random
import time

from bentoml.yatai.db.stores.lock import LockStore, LOCK_STATUS


class LockType:
    READ = LOCK_STATUS.read_lock
    WRITE = LOCK_STATUS.write_lock

class Lock():
    def __init__(self, db, lock_identifier, type=LockType.READ, timeout_seconds=10, timeout_jitter_seconds=1, max_retry_count=5):
        self.db = db
        self.id = lock_identifier
        self.type = type
        self.timeout = timeout_seconds
        self.jitter = timeout_jitter_seconds
        self.tries = max_retry_count

    def __enter__(self):
        for i in range(self.tries):
            try:
                with self.db.create_session() as sess:
                    ok = LockStore.acquire(sess, self.type, self.id)
                    if ok:
                        return sess
            except LockUnavailable as e:
                print(f"Failed to acquire lock: {e}. Retrying {self.tries - i} more times...")
                sleep_seconds = self.timeout + random.random() * self.jitter
                time.sleep(sleep_seconds)

    def __exit__(self, type, value, traceback):
        LockStore.release(self.id)