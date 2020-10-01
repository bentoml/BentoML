from bentoml.utils import cached_property


def test_cached_property():
    class A:
        def __init__(self):
            self.counter = 0

        @cached_property
        def a(self):
            self.counter += 1
            return "a"

    a = A()
    assert a.a == a.a
    assert a.counter == 1

    class B(A):
        @property
        def a(self):  # pylint: disable=invalid-overridden-method
            _a = super(B, self).a
            return _a

    b = B()
    assert b.a == b.a
    assert b.counter == 1

    class C(A):
        pass

    c = C()
    assert c.a == c.a
    assert c.counter == 1
