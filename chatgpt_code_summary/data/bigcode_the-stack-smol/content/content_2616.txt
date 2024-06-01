import asyncio
import discord
from discord.ext import commands
from discord.commands import slash_command, Option
import wavelink
import json
from dotenv import load_dotenv
import os

load_dotenv()

# Initiate json
file = open("config.json")
data = json.load(file)

# Public variables
guildID = data["guildID"][0]


class musicPlay(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_wavelink_node_ready(self, node: wavelink.Node):
        wavelink.NodePool.get_node(identifier=node.identifier)

    @commands.Cog.listener()
    async def on_wavelink_track_end(
        self, player: wavelink.Player, track: wavelink.Track, reason
    ):
        """When a track ends, check if there is another one in the queue."""
        await asyncio.sleep(5)
        if not player.queue.is_empty:
            next_track = player.queue.get()
            await player.play(next_track)

    @slash_command(guild_ids=[guildID], description="Play a song!")
    async def play(
        self, ctx, value: Option(str, required=True, description="Search for the song!")
    ):
        track = await wavelink.YouTubeTrack.search(query=value, return_first=True)
        if not ctx.user.voice:
            await ctx.respond("You must be in a voice channel to use music commands!")
        else:
            if not ctx.voice_client:
                vc: wavelink.Player = await ctx.author.voice.channel.connect(
                    cls=wavelink.Player
                )
            else:
                vc: wavelink.Player = ctx.voice_client

            if vc.is_playing():
                await vc.queue.put_wait(track)
                await ctx.respond(
                    f"{track.title} has been added to queue! Check the queue status using /queue!"
                )
            else:
                await vc.play(track)
                await ctx.respond(f"Now playing: {track.title}")

    @play.error
    async def play_error(self, ctx, error):
        await ctx.respond(f"`{error}`")


def setup(bot):
    bot.add_cog(musicPlay(bot))
