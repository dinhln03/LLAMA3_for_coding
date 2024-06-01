from nextcord.ext import commands
import requests

# the prefix is not used in this example
bot = commands.Bot(command_prefix='$')

# @bot.event
# async def on_message(message):
#     print(f'Message from {message.author}: {message.content}')

@bot.command()
async def ping(ctx):
    await ctx.send(f"The bot latency is {round(bot.latency * 1000)}ms.")

@bot.command()
async def greet(ctx):
    await ctx.send(f"Hello Master {ctx.author.mention}")

bot.run('your_token')