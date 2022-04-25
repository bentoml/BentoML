import signal
import subprocess
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    import typing as t


def kill_subprocess_tree(p: "subprocess.Popen[t.Any]") -> None:
    """
    Tell the process to terminate and kill all of its children. Availabe both on Windows and Linux.
    Note: It will return immediately rather than wait for the process to terminate.

    Args:
        p: subprocess.Popen object
    """
    if psutil.WINDOWS:
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(p.pid)])
    else:
        p.terminate()


def cancel_subprocess(p: "subprocess.Popen[t.Any]") -> None:
    if psutil.WINDOWS:
        p.send_signal(signal.CTRL_C_EVENT)  # type: ignore
    else:
        p.send_signal(signal.SIGINT)
