import io
import os
import random
import textwrap

from PIL import Image, ImageDraw, ImageFont
from telethon.tl.types import InputMessagesFilterDocument
from uniborg.util import admin_cmd


@borg.on(admin_cmd(pattern="srgb (.*)"))
async def sticklet(event):
    R = random.randint(0,256)
    G = random.randint(0,256)
    B = random.randint(0,256)
    
    sticktext = event.pattern_match.group(1)


    await event.delete()

    sticktext = textwrap.wrap(sticktext, width=10)

    sticktext = '\n'.join(sticktext)

    image = Image.new("RGBA", (512, 512), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    fontsize = 230

    FONT_FILE = await get_font_file(event.client, "@FontRes")

    font = ImageFont.truetype(FONT_FILE, size=fontsize)

    while draw.multiline_textsize(sticktext, font=font) > (512, 512):
        fontsize -= 3
        font = ImageFont.truetype(FONT_FILE, size=fontsize)

    width, height = draw.multiline_textsize(sticktext, font=font)
    draw.multiline_text(((512-width)/2,(512-height)/2), sticktext, font=font, fill=(R, G, B))

    image_stream = io.BytesIO()
    image_stream.name = "@AnonHexo.webp"
    image.save(image_stream, "WebP")
    image_stream.seek(0)


    await event.reply("https://t.me/AnonHexo", file=image_stream)


    try:
        os.remove(FONT_FILE)
    except:
        pass


async def get_font_file(client, channel_id):

    font_file_message_s = await client.get_messages(
        entity=channel_id,
        filter=InputMessagesFilterDocument,

        limit=None
    )
    font_file_message = random.choice(font_file_message_s)

    return await client.download_media(font_file_message)
