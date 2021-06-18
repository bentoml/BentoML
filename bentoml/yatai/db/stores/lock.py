import datetime
import enum

from sqlalchemy import UniqueConstraint, Column, Integer, Enum, DateTime, String
from bentoml.exceptions import LockUnavailable
from bentoml.yatai.db import Base


# LOCK_STATUS is an enum of the type of lock currently held
class LOCK_STATUS(enum.Enum):
    read_lock = 1
    write_lock = 2


class Lock(Base):
    __tablename__ = 'locks'
    __table_args__ = tuple(UniqueConstraint('resource_id', name='_resource_id_uc',))
    id = Column(Integer, primary_key=True)
    resource_id = Column(String, nullable=False, unique=True)
    lock_status = Column(Enum(LOCK_STATUS))
    locks_held = Column(Integer)
    ttl = Column(DateTime)

    # releases current lock
    def release(self, sess):
        self.locks_held -= 1

        # only delete lock row if no references left
        if self.locks_held == 0:
            sess.delete(self)

    # renews lock for `ttl_min` more min
    def renew(self, ttl_min):
        now = datetime.datetime.now()
        self.ttl = now + datetime.timedelta(minutes=ttl_min)


class LockStore(object):
    @staticmethod
    def _find_lock(sess, resource_id):
        lock_obj = sess.query(Lock).filter_by(resource_id=resource_id).first()
        return lock_obj

    # acquires a lock of type `lock_type` on `resource_id` that will expire in
    # `ttl_min` minutes
    @staticmethod
    def acquire(sess, lock_type, resource_id, ttl_min):
        now = datetime.datetime.now()
        ttl = now + datetime.timedelta(minutes=ttl_min)
        existing_lock = LockStore._find_lock(sess, resource_id)

        # no lock found; free to acquire
        if not existing_lock:
            # create lock with properties
            lock = Lock()
            lock.lock_status = lock_type
            lock.resource_id = resource_id
            lock.ttl = ttl
            lock.locks_held = 1
            sess.add(lock)
            return lock

        # existing lock expired, free to overwrite
        if existing_lock.ttl < now:
            existing_lock.lock_status = lock_type
            existing_lock.resource_id = resource_id
            existing_lock.ttl = ttl

            # locks_held can be held constant, we're just
            # overwriting existing lock
            return existing_lock

        # lock already exists
        if lock_type == LOCK_STATUS.read_lock:
            # acquire read lock
            if existing_lock.lock_status == LOCK_STATUS.write_lock:
                raise LockUnavailable("Failed to acquire read lock, write lock held")

            # current state is already read_lock, free to acquire read_lock
            # (multiple read_locks can be concurrently held)
            # bump ttl
            existing_lock.renew(ttl_min)
            existing_lock.locks_held += 1
            return existing_lock
        else:
            # acquire write lock
            # can't acquire write lock when any other lock is already held
            raise LockUnavailable("Failed to acquire write lock, another lock held")
