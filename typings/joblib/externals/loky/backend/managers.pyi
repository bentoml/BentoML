from multiprocessing.managers import SyncManager

class LokyManager(SyncManager):
    def start(self, initializer=..., initargs=...):  # -> None:
        """Spawn a server process for this manager object"""
        ...
