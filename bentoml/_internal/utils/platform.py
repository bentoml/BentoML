import subprocess

import psutil


def kill_subprocess_tree(p: "subprocess.Popen[bytes]") -> None:
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
