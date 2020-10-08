import atexit
from multiprocessing.util import _exit_function


def worker_exit(server, worker):  # pylint: disable=unused-argument
    from prometheus_client import multiprocess

    multiprocess.mark_process_dead(worker.pid)


def post_fork(server, worker):
    server.log.debug("Worker spawned (pid: %s)", worker.pid)


def pre_fork(server, worker):  # pylint: disable=unused-argument
    pass


def pre_exec(server):
    server.log.debug("Forked child, re-executing.")


def when_ready(server):
    server.log.debug("Server is ready. Spawning workers")


def worker_int(worker):
    worker.log.debug("worker received INT or QUIT signal")

    # get traceback info
    import threading
    import sys
    import traceback

    id2name = {th.ident: th.name for th in threading.enumerate()}
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" % (id2name.get(threadId, ""), threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    worker.log.debug("\n".join(code))


def worker_abort(worker):
    worker.log.debug("worker received SIGABRT signal")


def post_worker_init(worker):
    worker.log.debug('Unregistering usage tracking in worker process')
    atexit.unregister(_exit_function)  # Shutting down Gunicorn gracefully
