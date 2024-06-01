# @ayushk780
# Big Thanks To Spechide and @TechnoAyanBoT

"""Counth: Avaible commands: .bstats
"""
import asyncio
from telethon import events
from uniborg.util import admin_cmd, humanbytes,get_readable_time
import shutil
import time
from userbot import botStartTime


@borg.on(admin_cmd(pattern=r"bstats"))
async def _(event):
    if event.fwd_from:
        return
    
    currentTime = get_readable_time((time.time() - botStartTime))
    total, used, free = shutil.disk_usage('.')
    total = humanbytes(total)
    used = humanbytes(used)
    free = humanbytes(free)
    stats = f'Bot Uptime: {currentTime}\n' \
            f'Total disk space: {total}\n' \
                        f'Used: {used}\n' \
                        f'Free: {free}'

    await event.edit(str(stats))
