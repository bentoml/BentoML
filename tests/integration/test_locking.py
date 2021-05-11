import logging
import time
import threading
from tests.bento_service_examples.example_bento_service import ExampleBentoService
import subprocess

logger = logging.getLogger('bentoml.test')


class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)


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

    thread1 = ThreadWithResult(target=cli, args=(svc, 'containerize','-t','imagetag'))
    thread2 = ThreadWithResult(target=cli, args=(svc, 'delete', '-y',))

    thread1.start()
    time.sleep(1)
    thread2.start()

    thread1.join()
    thread2.join()
    containerize_output = "".join(list(thread1.result))
    delete_output = "".join(list(thread2.result))

    # make sure both commands run successfully
    print(containerize_output)
    print(delete_output)
    assert(f'Build container image: imagetag:{svc.version}' in containerize_output)
    assert("Failed to acquire write lock, another lock held. Retrying" in delete_output)
    assert(f"Deleted {svc.name}:{svc.version}" in delete_output)
