WIN32 = ...

def recursive_terminate(process, use_psutil=...): ...
def get_exitcodes_terminated_worker(processes):  # -> str:
    """Return a formated string with the exitcodes of terminated workers.

    If necessary, wait (up to .25s) for the system to correctly set the
    exitcode of one terminated worker.
    """
    ...
