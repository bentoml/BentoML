import datetime
import enum

from sqlalchemy import UniqueConstraint, Column, Integer, Enum, DateTime
from bentoml.exceptions import LockUnavailable
from bentoml.yatai.db import Base


class LOCK_STATUS(enum.Enum):
    read_lock = 1
    write_lock = 2


class Lock(Base):
    __tablename__ = 'locks'
    __table_args__ = tuple(UniqueConstraint('resource_id', name='_resource_id_uc',))
    id = Column(Integer, primary_key=True)
    resource_id = Column(Integer, nullable=False, unique=True)
    lock_status = Column(Enum(LOCK_STATUS))
    ttl = Column(DateTime)

    # releases current lock
    def release(self, sess):
        sess.delete(self)
        sess.commit()

    # renews lock for `ttl_min` more min
    def renew(self, sess, ttl_min):
        now = datetime.datetime.now()
        self.ttl = now + datetime.timedelta(minutes=ttl_min)
        sess.commit()

class LockStore(object):
    @staticmethod
    def _find_lock(sess, resource_id):
        lock_obj = sess.query(Lock).filter_by(resource_id=resource_id).first()
        return lock_obj

    @staticmethod
    def acquire(sess, lock_type, resource_id, ttl_min):
        now = datetime.datetime.now()
        ttl = now + datetime.timedelta(minutes=ttl_min)
        lock = LockStore._find_lock(sess, resource_id)

        # no lock found; free to acquire,
        # or existing lock is expired
        if not lock or lock.ttl < now:
            lock = Lock()
            lock.lock_status = lock_type
            lock.resource_id = resource_id
            lock.ttl = ttl
            sess.add(lock)
            sess.commit()
            return lock

        # acquire read lock
        if lock_type == LOCK_STATUS.read_lock:
            if lock.lock_status == LOCK_STATUS.write_lock:
                raise LockUnavailable("Failed to acquire read lock, write lock held")

            # read lock acquisition success, no change required
            # as current state is already read_lock

            # bump ttl
            lock.renew(sess, ttl_min)
            return lock
        else:
            # can't acquire read/write lock when any other lock is already held
            raise LockUnavailable("Failed to acquire write lock, another lock held")