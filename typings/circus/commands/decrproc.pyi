"""
This type stub file was generated by pyright.
"""

from circus.commands.incrproc import IncrProc

class DecrProc(IncrProc):
    """\
        Decrement the number of processes in a watcher
        ==============================================

        This comment decrement the number of processes in a watcher
        by <nbprocess>, 1 being the default.

        ZMQ Message
        -----------

        ::

            {
                "command": "decr",
                "propeties": {
                    "name": "<watchername>"
                    "nb": <nbprocess>
                    "waiting": False
                }
            }

        The response return the number of processes in the 'numprocesses`
        property::

            { "status": "ok", "numprocesses": <n>, "time", "timestamp" }

        Command line
        ------------

        ::

            $ circusctl decr <name> [<nb>] [--waiting]

        Options
        +++++++

        - <name>: name of the watcher
        - <nb>: the number of processes to remove.

    """
    name = ...
    properties = ...
    def execute(self, arbiter, props): # -> dict[str, Unknown | bool] | TransformableFuture:
        ...
    


