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
            self.status = self.STATUS_CLOSED
            outputs = await call(self.batch_input.values())
            self.batch_output = OrderedDict(
                [(k, v) for k, v in zip(self.batch_input.keys(), outputs)]
            )
            self.status = self.STATUS_CLOSED
            async with self.returned:
                self.returned.notify_all()
        except Exception as e:  # noqa TODO
            raise e
        finally:
            # make sure parade is closed
            self.status = self.STATUS_CLOSED


class ParadeDispatcher:
    def __init__(self, interval, loop=None):
        if loop is None:
            loop = asyncio.get_event_loop()
        self.loop = loop
        self.interval = interval
        self.callback = None
        self._current_parade = None

    def get_parade(self):
        if (self._current_parade
                and self._current_parade.status == Parade.STATUS_OPEN):
            return self._current_parade
        self._current_parade = Parade()
        self.loop.create_task(
            self._current_parade.start_wait(self.interval, self.callback))
        return self._current_parade

    async def __call__(self, callback):
        self.callback = callback

        async def _func(inputs):
            id_ = uuid.uuid4().hex
            parade = self.get_parade()
            parade.feed(id_, inputs)
            async with parade.returned:
                await parade.returned.wait()
            return parade.batch_output.get(id_)
        return _func


class MarshalService:
    def __init__(self, target_host="localhost", target_port=None):
        self.target_host = target_host
        self.target_port = target_port
        self.batch_handlers = dict()
    
    def set_target_port(self, target_port):
        self.target_port = target_port
    
    def add_batch_handler(self, api_name, batch_interval):
        if api_name not in self.batch_handlers:

            @ParadeDispatcher(batch_interval)
            async def _func(requests):
                reqs_s = await merge_aio_requests(requests)
                async with aiohttp.ClientSession() as client:
                    async with client.post(
                            f"http://{self.target_host}:{self.target_port}/{api_name}",
                            data=reqs_s) as resp:
                        resps = await split_aio_responses(resp)
                return resps

            self.batch_handlers[api_name] = _func

    async def request_handler(self, request):
        api_name = request.match_info['name']
        if api_name in self.batch_handlers:
            target_handler = self.get_target_handler(api_name)
            resp = await target_handler(request)
        else:
            async with aiohttp.ClientSession() as client:
                async with client.post(
                        f"http://{self.target_host}:{self.target_port}/{api_name}",
                        data=request.data,
                        headers=request.headers) as resp:
                    return resp
        return resp

    def make_app(self):
        app = aiohttp.web.Application()
        app.router.add_post('/{name}', self.request_handler)
        return app
