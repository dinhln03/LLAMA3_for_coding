from aiohttp import web
import asyncio
import uvloop

async def handle(request):
    name = request.match_info.get('name', "Anonymous")
    text = "Hello, " + name
    return web.Response(text=text)

app = web.Application()
app.add_routes([web.get('/', handle),
                web.get('/{name}', handle)])
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

web.run_app(app)
