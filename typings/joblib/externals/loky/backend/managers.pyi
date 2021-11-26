from multiprocessing.managers import SyncManager

class LokyManager(SyncManager):
    def start(self, initializer=None, initargs=()): ...
