import logging

from starlette.middleware import Middleware

logger = logging.getLogger(__name__)

class AccessLogMiddleware(Middleware):
    def __init__(self, app) -> None:
        self._app = app

    async def __call__(
        self, scope, receive, send
    ) -> None:
        logger.error("##############")
        await self._app(scope, receive, send)
