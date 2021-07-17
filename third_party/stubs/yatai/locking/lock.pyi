from typing import Any
from yatai.yatai.db.stores.lock import LOCK_STATUS as LOCK_STATUS, LockStore as LockStore

logger: Any
DEFAULT_TIMEOUT_SECONDS: int
DEFAULT_TIMEOUT_JITTER_SECONDS: int
DEFAULT_MAX_RETRY_COUNT: int
DEFAULT_TTL_MIN: int

class LockType:
    READ: Any
    WRITE: Any

def lock(db, locks, timeout_seconds=..., timeout_jitter_seconds=..., max_retry_count=..., ttl_min=...) -> None: ...
