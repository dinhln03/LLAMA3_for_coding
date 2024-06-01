import discord
from discord.ext import commands
import json


gamertags = 'gamer_tags.json'


class Gamertag(commands.Cog):

    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def newtag(self, ctx):
        with open(gamertags, 'r') as in_file:
            data = json.load(in_file)
        author_id = str(ctx.message.author.id)
        if author_id in list(data.keys()):
            await ctx.send('This user already has a gamertag library')
            print('command aborted: id already exists')
        else:
            data[author_id] = {}
            with open(gamertags, 'w') as out_file:
                json.dump(data, out_file)
            await ctx.send('Created new library under this user')

    @commands.command()
    async def addtag(self, ctx, platform, tag):
        with open(gamertags, 'r') as in_file:
            data = json.load(in_file)
        author_id = str(ctx.message.author.id)
        data[author_id][platform.lower()] = tag
        with open(gamertags, 'w') as out_file:
            json.dump(data, out_file)

    @commands.command()
    async def viewtag(self, ctx, mention, platform):
        with open(gamertags, 'r') as in_file:
            data = json.load(in_file)
        if platform.lower() == 'all':
            tags = f'{mention}\'s' + ' Gamer Tags'
            for plat, tag in data[str(ctx.message.mentions[0].id).lower()].items():
                tags += f'\n- {plat.capitalize()}: {tag}'
            await ctx.send(tags)
        else:
            await ctx.send(f'{mention}\'s {platform.capitalize()}: {data[str(ctx.message.mentions[0].id).lower()][platform.lower()]}')


def setup(bot):
    bot.add_cog(Gamertag(bot))
