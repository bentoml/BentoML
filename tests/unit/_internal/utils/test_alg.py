from bentoml._internal.utils.alg import FixedBucket
from bentoml._internal.utils.alg import TokenBucket


class TestFixedBucket:
    def test_empty(self):
        b = FixedBucket(3)
        assert len(b) == 0
        assert b.data == []
        assert b[:] == []

    def test_under_capacity(self):
        b = FixedBucket(3)
        b.put(1)
        b.put(2)
        assert len(b) == 2
        assert b.data == [1, 2]
        assert b[:] == [1, 2]

    def test_exactly_full(self):
        b = FixedBucket(3)
        for v in (1, 2, 3):
            b.put(v)
        assert len(b) == 3
        assert b[:] == [1, 2, 3]

    def test_wraps_and_iterates_fifo_oldest_first(self):
        b = FixedBucket(3)
        for v in (1, 2, 3, 4, 5):
            b.put(v)
        assert len(b) == 3
        # Indexing yields items oldest-to-newest after wrap-around.
        assert b[:] == [3, 4, 5]
        assert b[:2] == [3, 4]
        # data returns the raw ring-buffer contents (not reordered).
        assert b.data == [4, 5, 3]


class TestTokenBucket:
    def test_consume_within_amount_succeeds(self):
        bucket = TokenBucket(10)
        assert bucket.consume(5, avg_rate=0, burst_size=10) is True

    def test_consume_more_than_available_fails(self):
        bucket = TokenBucket(10)
        bucket.consume(5, avg_rate=0, burst_size=10)
        assert bucket.consume(10, avg_rate=0, burst_size=10) is False

    def test_burst_size_caps_available_tokens(self):
        bucket = TokenBucket(100)
        # burst_size caps usable tokens to 10, so a small take still succeeds.
        assert bucket.consume(1, avg_rate=0, burst_size=10) is True
        # ...but a take above the cap fails.
        assert bucket.consume(11, avg_rate=0, burst_size=10) is False
