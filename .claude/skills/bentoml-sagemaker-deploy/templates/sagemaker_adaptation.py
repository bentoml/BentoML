# SageMaker adaptation snippet for a BentoML v2 (SDK) service.
# Applied by the bentoml-sagemaker-deploy skill. Everything here is ADDITIVE:
# existing routes (/<method>, /readyz, /livez) keep working unchanged.
#
# Placeholders:
#   MyService        -> the user's @bentoml.service class
#   predict          -> the user's primary @bentoml.api method
#   (self, text: str) -> dict
#                    -> mirror the primary method's EXACT signature; the
#                       parameters define the JSON body SageMaker clients send.
#
# Validated with bentoml 1.4.39: `BENTOML_PORT=8080 bentoml serve .` then
#   GET  /ping        -> 200 {}
#   POST /invocations -> real inference JSON

# --- SageMaker adaptation (added by bentoml-sagemaker-deploy) ---
# Starlette ships with BentoML — no new dependency.
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

import bentoml


async def _sagemaker_ping(request):
    return JSONResponse({})


# IMPORTANT: an exact Route inside an app mounted at "/", NOT a mount at
# path="/ping" — a bare mount answers GET /ping with a 307 redirect to /ping/,
# which fails SageMaker's health check (it requires a literal 200).
_sagemaker_ping_app = Starlette(
    routes=[Route("/ping", _sagemaker_ping, methods=["GET"])]
)
# --- end SageMaker adaptation ---


@bentoml.asgi_app(
    _sagemaker_ping_app, path="/"
)  # SageMaker adaptation: GET /ping -> 200
@bentoml.service()  # <- the user's existing decorator, with its existing arguments
class MyService:
    @bentoml.api
    def predict(self, text: str) -> dict:  # <- the user's existing method, untouched
        ...

    # --- SageMaker adaptation: POST /invocations alias ---
    # Mirror the primary method's signature; delegate via `.local` (in-process call).
    # If the primary method is async:  async def invocations(...): return await self.predict.local(...)
    @bentoml.api(route="/invocations")
    def invocations(self, text: str) -> dict:
        return self.predict.local(text)

    # --- end SageMaker adaptation ---
