# based heavily on the example for starlette-auth-toolkit: https://github.com/florimondmanca/starlette-auth-toolkit

import typing

from starlette.authentication import requires
from starlette_authlib.middleware import AuthlibMiddleware as SessionMiddleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette_auth_toolkit.cryptography import PBKDF2Hasher
from starlette_auth_toolkit.base.backends import BaseBasicAuth

import bentoml
from bentoml.io import Text

# Password hasher
hasher = PBKDF2Hasher()


# Example user model
class User(typing.NamedTuple):
    username: str
    password: str


# Fake user storage
USERS = {
    "alice": User(username="alice", password=hasher.make_sync("alicepwd")),
    "bob": User(username="bob", password=hasher.make_sync("bobpwd")),
}


# Authentication backend
class BasicAuth(BaseBasicAuth):
    async def find_user(self, username: str):
        return USERS.get(username)

    async def verify_password(self, user: User, password: str):
        return await hasher.verify(password, user.password)


svc = bentoml.Service("auth_example", runners=[])


svc.add_asgi_middleware(
    AuthenticationMiddleware,
    backend=BasicAuth(),
    on_error=lambda _, exc: PlainTextResponse(str(exc), status_code=401),
)


@requires("authenticated")
def authenticate(request):
    return


@svc.api(input=Text(), output=Text())
def authenticated_echo(inp: str, ctx: bentoml.Context):
    print(dir(ctx.request))
    authenticate(ctx.request.http)
    return str
