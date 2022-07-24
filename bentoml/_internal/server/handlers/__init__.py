from __future__ import annotations

__all__ = ["Probe"]


class Probe:
    """
    A lightweight probe that can be used for both gRPC and HTTP protocols.
    """

    _is_ready: bool = False
    _is_live: bool = False

    def mark_as_ready(self):
        self._is_ready = True

    def mark_as_live(self):
        self._is_live = True

    def mark_as_dead(self):
        self._is_live = False
        self._is_ready = False

    @property
    def is_ready(self):
        return self._is_ready

    @property
    def is_live(self):
        return self._is_live
