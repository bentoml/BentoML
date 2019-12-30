# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections import OrderedDict
import asyncio
import logging
import uuid
import aiohttp

from bentoml import config
from bentoml.marshal.utils import merge_aio_requests, split_aio_responses


logger = logging.getLogger(__name__)


class Parade:
    STATUSES = (
        STATUS_OPEN,
        STATUS_CLOSED,
        STATUS_RETURNED,
    ) = range(3)

    def __init__(self):
        self.batch_input = OrderedDict()
        self.batch_output = None
        self.returned = asyncio.Condition()
        self.status = self.STATUS_OPEN

    def feed(self, id_, data):
        assert self.status == self.STATUS_OPEN
        self.batch_input[id_] = data
        return True

    async def start_wait(self, interval, call):
        try:
            await asyncio.sleep(interval)
            print(self.batch_input.keys())  # TODO: delete
            print(len(self.batch_input.keys()))  # TODO: delete
            self.status = self.STATUS_CLOSED
            outputs = await call(self.batch_input.values())
            self.batch_output = OrderedDict(
                [(k, v) for k, v in zip(self.batch_input.keys(), outputs)]
            )
            self.status = self.STATUS_RETURNED
            async with self.returned:
                self.returned.notify_all()
        except Exception as e:  # noqa TODO
            raise e
        finally:
            # make sure parade is closed
            self.status = self.STATUS_CLOSED


class ParadeDispatcher:
    def __init__(self, interval):
        self.interval = interval
        self.callback = None
        self._current_parade = None

    def get_parade(self):
        if (self._current_parade
                and self._current_parade.status == Parade.STATUS_OPEN):
            return self._current_parade
        self._current_parade = Parade()
        asyncio.get_event_loop().create_task(
            self._current_parade.start_wait(self.interval, self.callback))
        return self._current_parade

    def __call__(self, callback):
        self.callback = callback

        async def _func(inputs):
            id_ = uuid.uuid4().hex
            parade = self.get_parade()
            print(id_)  # TODO: delete
            parade.feed(id_, inputs)
            async with parade.returned:
                await parade.returned.wait()
            return parade.batch_output.get(id_)
        return _func


class MarshalService:
    _MARSHAL_FLAG = config("marshal_server").get("marshal_request_header_flag")

    def __init__(self, target_host="localhost", target_port=None):
        self.target_host = target_host
        self.target_port = target_port
        self.batch_handlers = dict()

    def set_target_port(self, target_port):
        self.target_port = target_port

    def add_batch_handler(self, api_name, max_latency):
        if api_name not in self.batch_handlers:

            @ParadeDispatcher(max_latency)  # TODO modify
            async def _func(requests):
                reqs_s = await merge_aio_requests(requests)
                async with aiohttp.ClientSession() as client:
                    async with client.post(
                            f"http://{self.target_host}:{self.target_port}/{api_name}",
                            data=reqs_s, headers={self._MARSHAL_FLAG: 'true'}) as resp:
                        resps = await split_aio_responses(resp)
                return resps

            self.batch_handlers[api_name] = _func

    async def request_handler(self, request):
        api_name = request.match_info['name']
        if api_name in self.batch_handlers:
            target_handler = self.batch_handlers[api_name]
            resp = await target_handler(request)
        else:
            data = await request.read()
            async with aiohttp.ClientSession() as client:
                async with client.post(
                        f"http://{self.target_host}:{self.target_port}/{api_name}",
                        data=data,
                        headers=request.headers) as resp:
                    body = await resp.read()
                    return aiohttp.web.Response(
                        status=resp.status,
                        body=body,
                        headers=resp.headers,
                    )
        return resp

    def make_app(self):
        app = aiohttp.web.Application()
        app.router.add_post('/{name}', self.request_handler)
        return app

    def fork_start_app(self, port):
        # Use new eventloop in the fork process to avoid problems on MacOS
        # ref: https://groups.google.com/forum/#!topic/python-tornado/DkXjSNPCzsI
        ev = asyncio.new_event_loop()
        asyncio.set_event_loop(ev)

        app = self.make_app()
        aiohttp.web.run_app(app, port=port)
