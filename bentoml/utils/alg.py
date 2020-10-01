import time


class FixedBucket:
    """
    Fixed size FIFO container.
    """

    def __init__(self, size):
        self._data = [None] * size
        self._cur = 0
        self._size = size
        self._flag_full = False

    def put(self, v):
        self._data[self._cur] = v
        self._cur += 1
        if self._cur == self._size:
            self._cur = 0
            self._flag_full = True

    @property
    def data(self):
        if not self._flag_full:
            return self._data[: self._cur]
        return self._data

    def __len__(self):
        if not self._flag_full:
            return self._cur
        return self._size

    def __getitem__(self, sl):
        if not self._flag_full:
            return self._data[: self._cur][sl]
        return (self._data[self._cur :] + self._data[: self._cur])[sl]


class TokenBucket:
    """
    Dynamic token bucket
    """

    def __init__(self, init_amount=0):
        self._amount = init_amount
        self._last_consume_time = time.time()

    def consume(self, take_amount, avg_rate, burst_size):
        now = time.time()
        inc = (now - self._last_consume_time) * avg_rate
        current_amount = min(inc + self._amount, burst_size)
        if take_amount > current_amount:
            return False
        self._amount, self._last_consume_time = current_amount - take_amount, now
        return True
