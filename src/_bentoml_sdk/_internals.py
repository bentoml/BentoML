from __future__ import annotations

import inspect
from typing import TYPE_CHECKING
from typing import Any

from bentoml.exceptions import BentoMLException

if TYPE_CHECKING:
    from fastapi import FastAPI


def make_fastapi_class_views(cls: type[Any], app: FastAPI) -> None:
    from fastapi import Depends
    from fastapi.routing import APIRoute
    from fastapi.routing import APIRouter
    from fastapi.routing import APIWebSocketRoute

    from .service import get_current_service

    class_methods = [meth for meth in vars(cls).values() if callable(meth)]
    api_routes = [
        route
        for route in app.routes
        if isinstance(route, (APIRoute, APIWebSocketRoute))
        and route.endpoint in class_methods
    ]
    if not api_routes:
        return
    # Modify these routes and mount it to a new APIRouter.
    # We need to to this (instead of modifying in place) because we want to use
    # the app.include_router to re-run the dependency analysis for each routes.
    new_router = APIRouter()
    for route in api_routes:
        app.routes.remove(route)
        # Get the original endpoint and add a `Depends` to the `self` parameter
        signature = inspect.signature(route.endpoint)
        parameters = list(signature.parameters.values())
        if not parameters:
            raise BentoMLException(
                "FastAPI class views must have at least one parameter"
            )
        self_param = parameters[0].replace(default=Depends(get_current_service))
        new_params = [self_param] + [
            # Convert all parameters to keyword-only since the first param
            # is no longer positional
            param.replace(kind=inspect.Parameter.KEYWORD_ONLY)
            for param in parameters[1:]
        ]
        new_signature = signature.replace(parameters=new_params)
        setattr(route.endpoint, "__signature__", new_signature)
        new_router.routes.append(route)
    app.include_router(new_router)
