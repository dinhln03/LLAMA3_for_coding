import asyncio, logging
from aiohttp import web


logging.basicConfig(level=logging.INFO)


def index(request):
    return web.Response(body=b'<h1>Hello World</h1>', content_type='text/html')


async def init(loop):
    app = web.Application(loop=loop)
    app.router.add_route('GET', '/', index)
    srv = await loop.create_server(app.make_handler(), '127.0.0.1', 5000)
    logging.info('server is listening at http://127.0.0.1:5000')
    return srv

cur_loop = asyncio.get_event_loop()
cur_loop.run_until_complete(init(cur_loop))
cur_loop.run_forever()



