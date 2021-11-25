from sys import version_info

_deprecated_names = ...
if version_info[:2] >= (3, 7):
    def __getattr__(name): ...

else: ...

class WorkerInterrupt(Exception):
    """An exception that is not KeyboardInterrupt to allow subprocesses
    to be interrupted.
    """

    ...
