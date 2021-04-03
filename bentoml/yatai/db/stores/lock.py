import enum
import re

from sqlalchemy import UniqueConstraint, Column, Integer, String, and_, Enum

from bentoml.exceptions import YataiLabelException, InvalidArgument, LockUnavailable
from bentoml.yatai.db import Base
from bentoml.yatai.proto.label_selectors_pb2 import LabelSelectors


class LOCK_STATUS(enum.Enum):
    read_lock = 1
    write_lock = 2


class Lock(Base):
    __tablename__ = 'locks'
    __table_args__ = tuple(
        UniqueConstraint(
            'resource_id',
            name='_resource_id_uc',
        )
    )
    id = Column(Integer, primary_key=True)
    resource_id = Column(Integer, nullable=False)
    lock_status = Column(Enum(LOCK_STATUS))


class LockStore(object):
    @staticmethod
    def _find_lock(sess, resource_id):
        lock_obj = (
            sess.query(Lock)
            .filter_by(resource_id=resource_id)
            .first()
        )
        return lock_obj

    @staticmethod
    def acquire(sess, lock_type, resource_id):
        lock = LockStore._find_lock(sess, resource_id)

        # no lock found; free to acquire
        if not lock:
            lock = Lock()
            lock.lock_status = lock_type
            lock.resource_id = resource_id
            sess.add(lock)
            return True

        # acquire read lock
        if lock_type == LOCK_STATUS.read_lock:
            if lock.lock_status == LOCK_STATUS.write_lock:
                raise LockUnavailable("Failed to acquire read lock, write lock held")

            # read lock acquisition success, no change required
            # as current state is already read_lock
            return True
        else:
            # can't acquire read/write lock when write lock is already held
            raise LockUnavailable(f"Failed to acquire write lock, lock held")

    @staticmethod
    def release(sess, resource_id):
        lock = LockStore._find_lock(sess, resource_id)
        sess.delete(lock)
