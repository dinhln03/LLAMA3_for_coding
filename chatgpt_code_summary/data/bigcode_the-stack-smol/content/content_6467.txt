import asyncio

import discord
from discord.ext.commands import Bot
from discord.ext import commands
from discord import Color, Embed

import backend.commands as db
from backend import strikechannel


# This command allows players to change their name.
#
# !name [new_name]
#
# This replaces the default nickname changing that Discord provides so
# that their name will also be replaced in the spreadsheet.
class Name(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.strike_channel_id = strikechannel


    @commands.command()
    async def name(self, ctx):
        old_name = ctx.author.display_name
        new_name = ctx.message.content[6:]

        print(old_name)
        print(new_name)

        # This changes their name in the "#strikes" channel
        channel = self.bot.get_channel(self.strike_channel_id)
        async for msg in channel.history(limit=None):
            text = msg.content.replace("```", "")
            text_lst = text.split("\n")

            d = {}
            for line in text_lst:
                try:
                    name, strikes = line.rsplit(" - ", 1)
                except:
                    continue
                d[name] = int(strikes)

            if old_name in d:
                d[new_name] = d[old_name]
                del d[old_name]

            inner_text = ""
            for k, v in d.items():
                inner_text += f"{k} - {v}\n"


            full_text = f"```\n{inner_text}```"
            await msg.edit(content=full_text)

        db.change_name(old_name, new_name)

        await ctx.author.edit(nick=new_name)
        await ctx.channel.send("Name Changed!")



def setup(bot):
    bot.add_cog(Name(bot))
