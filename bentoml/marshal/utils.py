import pickle
import asyncio


async def merge_aio_requests(reqs) -> bytes:
    body_list = await asyncio.gather(*tuple(req.read() for req in reqs))
    headers_list = tuple(req.raw_headers for req in reqs)
    merged_reqs = [dict(headers=h, data=b) for h, b in zip(headers_list, body_list)]
    return pickle.dumps(merged_reqs)


async def split_aio_responses(ori_response):
    from aiohttp import web
    merged = await ori_response.read()
    try:
        merged_responses = pickle.loads(merged)
    except pickle.UnpicklingError:
        raise  # TODO catch

    if ori_response.status != 200:
        return [web.Response(status=ori_response.status) for _ in merged_responses]
    return [
        web.Response(status=ori_response.status, body=i['data'], headers=i['headers'])
        for i in merged_responses
    ]


def split_flask_requests(req):
    import flask
    raw = req.data
    info_list = pickle.loads(raw)
    return [
        flask.Request.from_values(
            path=req.path,
            base_url=req.base_url,
            query_string=req.query_string,
            method=req.method,
            headers=i['headers'],
            data=i['data']
        ) for i in info_list]


def merge_flask_responses(resps) -> bytes:
    merged_resps = [dict(headers=tuple(r.headers), data=r.data) for r in resps]
    return pickle.dumps(merged_resps)
