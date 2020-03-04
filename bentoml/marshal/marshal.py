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

import asyncio
import logging
from typing import Callable
from functools import lru_cache, partial

import aiohttp
from bentoml import config
from bentoml.utils.trace import async_trace, make_http_headers
from bentoml.marshal.utils import DataLoader, SimpleRequest


logger = logging.getLogger(__name__)
ZIPKIN_API_URL = config("tracing").get("zipkin_api_url")


class Parade:
    STATUSES = (STATUS_OPEN, STATUS_CLOSED, STATUS_RETURNED,) = range(3)

    def __init__(self, max_size, outbound_sema):
        self.max_size = max_size
        self.outbound_sema = outbound_sema
        self.batch_input = [None] * max_size
        self.batch_output = [None] * max_size
        self.cur = 0
        self.returned = asyncio.Condition()
        self.status = self.STATUS_OPEN

    def feed(self, data) -> Callable:
        '''
        feed data into this parade.
        return:
            the output index in parade.batch_output
        '''
        assert self.status == self.STATUS_OPEN
        self.batch_input[self.cur] = data
        self.cur += 1
        if self.cur == self.max_size:
            self.status = self.STATUS_CLOSED
        return self.cur - 1

    async def start_wait(self, interval, call):
        with async_trace(
            ZIPKIN_API_URL,
            service_name=self.__class__.__name__,
            span_name="[1]parade task",
            sample_rate=1,
            is_root=True,
        ):
            try:
                # await asyncio.sleep(interval)
                await asyncio.sleep(0.02)
                with async_trace(
                    ZIPKIN_API_URL,
                    service_name=self.__class__.__name__,
                    span_name="[2]call",
                ):
                    async with self.outbound_sema:
                        self.status = self.STATUS_CLOSED
                        self.batch_output = await call(self.batch_input[: self.cur])
                        self.status = self.STATUS_RETURNED
                        async with self.returned:
                            self.returned.notify_all()
            except Exception as e:  # noqa TODO
                raise e
            finally:
                # make sure parade is closed
                if self.status != self.STATUS_OPEN:
                    self.status = self.STATUS_CLOSED


class ParadeDispatcher:
    def __init__(self, interval, max_size):
        """
        params:
            * interval: milliseconds
        """
        self.interval = interval
        self.max_size = max_size
        self.callback = None
        self._current_parade = None

    @property
    @lru_cache(maxsize=1)
    def outbound_sema(self):
        '''
        create semaphore lazily
        '''
        # TODO(hrmthw): config
        return asyncio.Semaphore(3)

    def get_parade(self):
        if self._current_parade and self._current_parade.status == Parade.STATUS_OPEN:
            return self._current_parade
        self._current_parade = Parade(self.max_size, self.outbound_sema)
        asyncio.get_event_loop().create_task(
            self._current_parade.start_wait(self.interval / 1000.0, self.callback)
        )
        return self._current_parade

    def __call__(self, callback):
        self.callback = callback

        async def _func(inputs):
            parade = self.get_parade()
            _id = parade.feed(inputs)
            async with parade.returned:
                await parade.returned.wait()
            return parade.batch_output[_id]

        return _func


class MarshalService:
    _MARSHAL_FLAG = config("marshal_server").get("marshal_request_header_flag")

    def __init__(self, target_host="localhost", target_port=None):
        self.target_host = target_host
        self.target_port = target_port
        self.batch_handlers = dict()

    def set_target_port(self, target_port):
        self.target_port = target_port

    def add_batch_handler(self, api_name, max_latency, max_batch_size):
        if api_name not in self.batch_handlers:
            _func = ParadeDispatcher(max_latency, max_batch_size)(
                partial(self._batch_handler_template, api_name=api_name)
            )
            self.batch_handlers[api_name] = _func

    async def request_dispatcher(self, request):
        with async_trace(
            ZIPKIN_API_URL,
            service_name=self.__class__.__name__,
            span_name=f"[1]http request",
            is_root=True,
            standalone=True,
            sample_rate=0.001,
        ):
            api_name = request.match_info["name"]
            if api_name in self.batch_handlers:
                req = SimpleRequest(await request.read(), request.raw_headers)
                resp = await self.batch_handlers[api_name](req)
                return resp
            else:
                resp = await self._relay_handler(request, api_name)
                return resp

    def make_app(self):
        app = aiohttp.web.Application()
        app.router.add_view("/{name}", self.request_dispatcher)
        return app

    def fork_start_app(self, port):
        # Use new eventloop in the fork process to avoid problems on MacOS
        # ref: https://groups.google.com/forum/#!topic/python-tornado/DkXjSNPCzsI
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app = self.make_app()
        aiohttp.web.run_app(app, port=port)

    async def _relay_handler(self, request, api_name):
        data = await request.read()
        headers = dict(request.headers)
        api_url = f"http://{self.target_host}:{self.target_port}/{api_name}"

        with async_trace(
            ZIPKIN_API_URL,
            service_name=self.__class__.__name__,
            span_name=f"[2]{api_name} relay",
        ) as trace_ctx:
            headers.update(make_http_headers(trace_ctx))
            async with aiohttp.ClientSession() as client:
                async with client.request(
                    request.method, api_url, data=data, headers=request.headers
                ) as resp:
                    body = await resp.read()
        return aiohttp.web.Response(
            status=resp.status, body=body, headers=resp.headers,
        )

    async def _batch_handler_template(self, requests, api_name):
        headers = {self._MARSHAL_FLAG: "true"}
        api_url = f"http://{self.target_host}:{self.target_port}/{api_name}"

        with async_trace(
            ZIPKIN_API_URL,
            service_name=self.__class__.__name__,
            span_name=f"[2]merged {api_name}",
        ) as trace_ctx:
            headers.update(make_http_headers(trace_ctx))
            reqs_s = DataLoader.merge_aio_requests(requests)
            async with aiohttp.ClientSession() as client:
                async with client.post(api_url, data=reqs_s, headers=headers) as resp:
                    raw = await resp.read()

        merged = DataLoader.split_aio_responses(raw)
        if merged is None:
            return (aiohttp.web.HTTPInternalServerError,) * len(requests)
        return tuple(
            aiohttp.web.Response(body=i.data, headers=i.headers, status=i.status)
            for i in merged
        )
