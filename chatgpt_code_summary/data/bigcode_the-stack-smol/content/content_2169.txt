import discord
import io
import aiohttp
from aiohttp import request, ClientSession
from src.embeds.image_embed import ImageEmbed


async def request_canvas_image(ctx, url, member: discord.Member = None, params={}, is_gif=False):
    params_url = "&" + "&".join(["{}={}".format(k, v)
                                for k, v in params.items()]) if params != {} else ""
    async with ClientSession() as wastedSession:
        async with wastedSession.get(f'{url}?avatar={member.avatar_url_as(format="png", size=1024)}{params_url}') as wastedImage:
            imageData = io.BytesIO(await wastedImage.read())
            await wastedSession.close()
            await ctx.send(file=discord.File(imageData, 'image.gif' if is_gif else 'image.png'))


async def request_image(ctx, url, params={}, key="link", title="Requested image", description="", footer=""):
    params_url = "&" + "&".join(["{}={}".format(k, v)
                                for k, v in params.items()]) if params != {} else ""
    async with request("GET", url + params_url) as response:
        if response.status == 200:
            data = await response.json()
            if "caption" in data:
                title = data["caption"]

            embed = ImageEmbed(
                ctx,
                title,
                description,
                footer,
                data[key]
            )
            await embed.send()
        else:
            await ctx.send(f"API returned a {response.status} status :((")


async def request_text(ctx, url, key, params={}, text_format="{}"):
    params_url = "&" + "&".join(["{}={}".format(k, v)
                                for k, v in params.items()]) if params != {} else ""
    print(params_url)
    async with request("GET", url + params_url) as response:
        if response.status == 200:
            data = await response.json()
            print(data)
            await ctx.send(text_format.format(data[key]))
        else:
            await ctx.send(f"API returned a {response.status} status :((")
