import logging; logging.basicConfig(level=logging.INFO)

import asyncio, os, json, time
from datetime import datetime

from aiohttp import web
import aiomysql


async def index(request):
    return web.Response(text='Awsome')


app = web.Application()
app.add_routes([web.get('/', index)])
logging.info('server started at http://127.0.0.1:9000 ...')
web.run_app(app, host='127.0.0.1', port=9000)
