import enum
import re

from sqlalchemy import UniqueConstraint, Column, Integer, String, and_, Enum

from bentoml.exceptions import YataiLabelException, InvalidArgument, LockUnavailable
from bentoml.yatai.db import Base
from bentoml.yatai.proto.label_selectors_pb2 import LabelSelectors


class RESOURCE_TYPE(enum.Enum):
    deployment = 1
    bento = 2

class LOCK_STATUS(enum.Enum):
    read_lock = 1
    write_lock = 2


class Lock(Base):
    __tablename__ = 'locks'
    __table_args__ = tuple(
        UniqueConstraint(
            'resource_type',
            'resource_id',
            name='_resource_type_resource_id_uc',
        )
    )
    id = Column(Integer, primary_key=True)
    resource_type = Column(Enum(RESOURCE_TYPE))
    resource_id = Column(Integer, nullable=False)
    lock_status = Column(Enum(LOCK_STATUS))


class LockStore(object):
    @staticmethod
    def _find_lock(sess, resource_type, resource_id):
        lock_obj = (
            sess.query(Lock)
            .filter_by(
                resource_id=resource_id,
                resource_type=resource_type,
            )
            .one()
        )
        return lock_obj

    @staticmethod
    def acquire(sess, lock, type, resource_type, resource_id):
        # no lock found; free to acquire
        if not lock:
            lock = Lock()
            lock.lock_status = type
            lock.resource_id = resource_id
            lock.resource_type = resource_type
            sess.add(lock)
            return True

        # acquire read lock
        if type == LOCK_STATUS.read_lock:
            if lock.lock_status == LOCK_STATUS.write_lock:
                raise LockUnavailable("Failed to acquire read lock, write lock held")

            # read lock acquisition success, no change required
            # as current state is already read_lock
            return True
        else:
            # can't acquire read/write lock when write lock is already held
            raise LockUnavailable(f"Failed to acquire write lock, lock held")

    # thin wrapper around acquire
    @staticmethod
    def acquire_read(sess, resource_type, resource_id):
        lock = LockStore._find_lock(sess, resource_type, resource_id)
        LockStore.acquire(sess, lock, LOCK_STATUS.read_lock)

    @staticmethod
    def acquire_write(sess, resource_type, resource_id):
        lock = LockStore._find_lock(sess, resource_type, resource_id)
        LockStore.acquire(sess, lock, LOCK_STATUS.write_lock)

    @staticmethod
    def release(sess, resource_type, resource_id):
        lock = LockStore._find_lock(sess, resource_type, resource_id)
        sess.delete(lock)
