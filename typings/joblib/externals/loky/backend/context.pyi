import sys
from multiprocessing.context import BaseContext

from .process import LokyInitMainProcess, LokyProcess

START_METHODS = ...
_DEFAULT_START_METHOD = ...
physical_cores_cache = ...
if sys.version_info[:2] >= (3, 4):
    def get_context(method=...): ...

else:
    BaseContext = ...
    def get_spawning_popen(): ...
    def set_spawning_popen(popen): ...
    def assert_spawning(obj): ...
    def get_context(method=...): ...

def set_start_method(method, force=...): ...
def get_start_method(): ...
def cpu_count(only_physical_cores=...):  # -> int:
    """Return the number of CPUs the current process can use.

    The returned number of CPUs accounts for:
     * the number of CPUs in the system, as given by
       ``multiprocessing.cpu_count``;
     * the CPU affinity settings of the current process
       (available with Python 3.4+ on some Unix systems);
     * CFS scheduler CPU bandwidth limit (available on Linux only, typically
       set by docker and similar container orchestration systems);
     * the value of the LOKY_MAX_CPU_COUNT environment variable if defined.
    and is given as the minimum of these constraints.

    If ``only_physical_cores`` is True, return the number of physical cores
    instead of the number of logical cores (hyperthreading / SMT). Note that
    this option is not enforced if the number of usable cores is controlled in
    any other way such as: process affinity, restricting CFS scheduler policy
    or the LOKY_MAX_CPU_COUNT environment variable. If the number of physical
    cores is not found, return the number of logical cores.

    It is also always larger or equal to 1.
    """
    ...

class LokyContext(BaseContext):
    """Context relying on the LokyProcess."""

    _name = ...
    Process = LokyProcess
    cpu_count = ...
    def Queue(self, maxsize=..., reducers=...):  # -> Queue:
        """Returns a queue object"""
        ...
    def SimpleQueue(self, reducers=...):  # -> SimpleQueue:
        """Returns a queue object"""
        ...
    if sys.version_info[:2] < (3, 4):
        def get_context(self): ...
        def get_start_method(self): ...
        def Pipe(self, duplex=...):  # -> tuple[Connection, Connection]:
            """Returns two connection object connected by a pipe"""
            ...
    if sys.platform != "win32":
        def Semaphore(self, value=...):  # -> Semaphore:
            """Returns a semaphore object"""
            ...
        def BoundedSemaphore(self, value):  # -> BoundedSemaphore:
            """Returns a bounded semaphore object"""
            ...
        def Lock(self):  # -> Lock:
            """Returns a lock object"""
            ...
        def RLock(self):  # -> RLock:
            """Returns a recurrent lock object"""
            ...
        def Condition(self, lock=...):  # -> Condition:
            """Returns a condition object"""
            ...
        def Event(self):  # -> Event:
            """Returns an event object"""
            ...

class LokyInitMainContext(LokyContext):
    """Extra context with LokyProcess, which does load the main module

    This context is used for compatibility in the case ``cloudpickle`` is not
    present on the running system. This permits to load functions defined in
    the ``main`` module, using proper safeguards. The declaration of the
    ``executor`` should be protected by ``if __name__ == "__main__":`` and the
    functions and variable used from main should be out of this block.

    This mimics the default behavior of multiprocessing under Windows and the
    behavior of the ``spawn`` start method on a posix system for python3.4+.
    For more details, see the end of the following section of python doc
    https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
    """

    _name = ...
    Process = LokyInitMainProcess

if sys.version_info > (3, 4):
    ctx_loky = ...
