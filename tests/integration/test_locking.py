import logging
import time
import threading
from tests.bento_service_examples.example_bento_service import ExampleBentoService
import concurrent.futures
import subprocess

class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)

logger = logging.getLogger('bentoml.test')

def cli(svc, cmd, *args):
    bento_tag = f'{svc.name}:{svc.version}'
    proc = subprocess.Popen(
        [
            'bentoml',
            cmd,
            bento_tag,
            *args,
        ]
        , stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout = proc.stdout.read().decode('utf-8')
    stderr = proc.stderr.read().decode('utf-8')
    return stdout, stderr

def test_lock():
    svc = ExampleBentoService()
    svc.pack('model', [1, 2, 3])
    svc.save()

    thread1 = ThreadWithResult(target=cli, args=(svc, 'containerize',))
    thread2 = ThreadWithResult(target=cli, args=(svc, 'delete', '-y',))

    start = time.time()
    thread1.start()
    time.sleep(5)
    thread2.start()

    thread1.join()
    thread2.join()
    end = time.time()
    print(f">>> thread {end-start}s elapsed")
    print("$ containerize")
    for l in thread1.result:
        print(l)
    print("$ delete")
    for l in thread2.result:
        print(l)
    assert(1 == 2)