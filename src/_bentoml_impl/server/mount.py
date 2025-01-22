from __future__ import annotations

from starlette.applications import Starlette
from starlette.routing import Match
from starlette.routing import Mount
from starlette.types import Scope


class PassiveMount(Mount):
    """A subclass of mount that doesn't match the path prefix eagerly."""

    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        match, child_scope = super().matches(scope)
        if match == Match.FULL and isinstance(self.app, Starlette):
            scope = {**scope, **child_scope}
            partial_match: Match | None = None
            for route in self.app.routes:
                child_match, _ = route.matches(scope)
                if child_match == Match.FULL:
                    return child_match, child_scope
                if child_match == Match.PARTIAL and partial_match is None:
                    partial_match = child_match

            if partial_match is not None:
                return partial_match, child_scope
            return Match.NONE, child_scope
        return match, child_scope
