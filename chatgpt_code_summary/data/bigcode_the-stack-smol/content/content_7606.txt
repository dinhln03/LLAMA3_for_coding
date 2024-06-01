#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nonebot
from nonebot.adapters.cqhttp import Bot as CQHTTPBot, message

# 初始化nb
nonebot.init()

# 连接驱动
driver = nonebot.get_driver()
driver.register_adapter("cqhttp", CQHTTPBot)

# 加载插件(除此处其他配置不建议更改)
nonebot.load_builtin_plugins()
nonebot.load_plugins('src/plugins')
nonebot.load_plugin('nonebot_plugin_help')
nonebot.load_plugin('nonebot_plugin_apscheduler')

app = nonebot.get_asgi()


from nonebot import Bot
@driver.on_bot_connect
async def do_something(bot: Bot):
    from utils import NAME, ONWER
    await bot.send_private_msg(
        user_id=ONWER,
        message=f'{NAME} 已启动'
    )
if __name__ == "__main__":
    nonebot.run()
