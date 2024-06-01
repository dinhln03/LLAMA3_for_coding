# IMPORT MANAGEMENT
try:
    import gevent.monkey
except ModuleNotFoundError:
    import os
    os.system('pip install -r requirements.txt')
    import gevent.monkey

gevent.monkey.patch_all() # patch everything

import colorama
colorama.init(autoreset=True)

import discord.commands
import asyncio
import discord
import dotenv
import os

# IMPORTS
from discord.ext import commands
from cogs.helpers import config, management
from discord_together import DiscordTogether


# SETTINGS
COLOR = config.load()['color-primary']
TESTING_MODE = management.testing_mode()
PREFIX = '//'

# SETUP
dotenv.load_dotenv()  # initialize virtual environment
token = os.getenv('DISCORD_TOKEN')
client = commands.Bot(command_prefix=PREFIX, intents=discord.Intents.all())

async def status_task():
    while True:
        await client.change_presence(activity=discord.Game(f'v0.5・open source'))

@client.event
async def on_ready():
    management.set_start_time()

    if management.testing_mode():
        await client.change_presence(status=discord.Status.idle)

    print(colorama.Fore.GREEN + 'ONLINE as', client.user)

    client.togetherControl = await DiscordTogether(token)
    client.loop.create_task(status_task())

# load cogs
# credit: https://youtu.be/vQw8cFfZPx0
for filename in os.listdir(os.getcwd() + '/src/cogs/'):
    if filename.endswith('.py'):
        client.load_extension(f'cogs.{filename[:-3]}')

try:
    client.run(token) # run bot with the token set in the .env file
except:
    print(colorama.Fore.RED + 'Unable to run the client. Please check your bot token.')