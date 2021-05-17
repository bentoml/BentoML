import time
import threading


class ThreadWithResult(threading.Thread):
    def __init__(
            self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None
    ):
        def function():
            self.result = target(*args, **kwargs)

        super().__init__(group=group, target=function, name=name, daemon=daemon)


def run_delayed_thread(t1, t2, delay=1):
    t1.start()
    time.sleep(delay)
    t2.start()

    t1.join()
    t2.join()
