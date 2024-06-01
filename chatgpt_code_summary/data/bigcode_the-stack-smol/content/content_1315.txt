import uvloop
import asyncio
import jinja2
import aiohttp_jinja2
from aiohttp import web
from quicksets import settings

from app.middlewares import middlewares
from app.views import routes


async def create_app():
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    app = web.Application(middlewares=middlewares)
    aiohttp_jinja2.setup(
        app, loader=jinja2.FileSystemLoader(settings.TEMPLATES_PATH))
    app.add_routes(routes)
    return app


if __name__ == '__main__':
    app = create_app()
    web.run_app(app, host=settings.HOST, port=settings.PORT)
