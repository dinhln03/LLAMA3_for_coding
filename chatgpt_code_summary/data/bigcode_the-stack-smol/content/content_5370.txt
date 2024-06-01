import discord
from discord.ext import commands
import humanize
import traceback
import random
import datetime
import json

class Information(commands.Cog):
    def __init__(self, client):
        self.client = client
        self.launched_at = datetime.datetime.utcnow()

    @commands.Cog.listener()
    async def on_ready(self):
        print("Information is ready")

    @commands.command(aliases = ["guild", "guildinfo", "si"])
    async def serverinfo(self, ctx):

        findbots = sum(1 for member in ctx.guild.members if member.bot)
        roles = sum(1 for role in ctx.guild.roles)

        embed = discord.Embed(title = 'Infomation about ' + ctx.guild.name + '.', colour = ctx.author.color)
        embed.set_thumbnail(url = str(ctx.guild.icon_url))
        embed.add_field(name = "Guild's name: ", value = ctx.guild.name)
        embed.add_field(name = "Guild's owner: ", value = str(ctx.guild.owner))
        embed.add_field(name = "Guild's verification level: ", value = str(ctx.guild.verification_level))
        embed.add_field(name = "Guild's id: ", value = f"`{ctx.guild.id}`")
        embed.add_field(name = "Guild's member count: ", value = f"{ctx.guild.member_count}")
        embed.add_field(name="Bots", value=f"`{findbots}`", inline=True)
        embed.add_field(name = "Guild created at: ", value = str(ctx.guild.created_at.strftime("%a, %d %B %Y, %I:%M %p UTC")))
        embed.add_field(name = "Number of Roles:", value = f"`{roles}`")
        embed.set_footer(text='Bot Made by NightZan999#0194')
        embed.set_author(name = ctx.author.name, icon_url = ctx.author.avatar_url)

        await ctx.send(embed =  embed)

    @commands.command(aliases = ["ci"])
    async def channelinfo(self, ctx, channel : discord.TextChannel = None):
        if channel == None:
            channel = ctx.channel
        
        em = discord.Embed(title = f"Info about {channel.name}", color = ctx.author.color, description = f"Here is an insight into {channel.mention}")
        em.add_field(name = "ID:", value = f"`{channel.id}`")
        em.add_field(name = "Name:", value = f"`{channel.name}`")
        em.add_field(name = "Server it belongs to:", value = f"{channel.guild.name}", inline = True)
        
        try:
            em.add_field(name = "Category ID:", value = f"`{channel.category_id}`", inline = False)
        except:
            pass
        em.add_field(name = "Topic:", value = f"`{channel.topic}`")
        em.add_field(name = "Slowmode:", value = f"`{channel.slowmode_delay}`", inline = True)

        em.add_field(name = "People who can see the channel:", value = f"`{len(channel.members)}`", inline = False)
        em.add_field(name = "Is NSFW:", value = f"`{channel.is_nsfw()}`")
        em.add_field(name = "Is News:", value = f"`{channel.is_news()}`", inline = True)
        
        em.set_footer(text = "invite me ;)", icon_url = ctx.author.avatar_url)
        em.set_thumbnail(url = str(ctx.guild.icon_url))
        em.set_author(name = ctx.author.name, icon_url = ctx.author.avatar_url)
        
        await ctx.send(embed = em)

    @commands.command()
    async def userinfo(self, ctx, member : discord.Member = None):
        if member == None:
            member = ctx.author
        pos = sum(m.joined_at < member.joined_at for m in ctx.guild.members if m.joined_at is not None)
        roles = [role for role in member.roles]
        embed = discord.Embed(title = "👨 Info", color = discord.Color.random(), description = f"Information about: {member.name}")
        embed.add_field(name = "Nickname", value = member.nick or None)
        embed.add_field(name = "Verification Pending", value = member.pending)
        embed.add_field(name = "Status:", value = member.raw_status)
        if member.mobile_status:
            device = "Mobile"
        elif member.desktop_status:
            device = "Desktop"
        elif member.web_status:
            device=  "Web"
        embed.add_field(name = "Discord Device:", value = device)
        embed.add_field(name = "Color", value = member.color)
        embed.add_field(name = "Mention:", value = member.mention)
        embed.add_field(name = "Top Role:", value = member.top_role.mention)
        embed.add_field(name = "Voice State:", value = member.voice or None)
        embed.set_footer(icon_url=member.avatar_url, text=f'Requested By: {ctx.author.name}')
        await ctx.send(embed=embed)
    
    @userinfo.error
    async def userinfo_error(self, ctx, error):
        if isinstance(error, commands.BadArgument):
            em = discord.Embed(title = f"<:fail:761292267360485378> Userinfo Error", color = ctx.author.color)
            em.add_field(name = f"Reason:", value = f"Arguments were of the wrong data type!")
            em.add_field(name = "Args", value = "```diff\n+ imp userinfo <user>\n- imp userinfo e\n```")
            em.set_thumbnail(url = ctx.author.avatar_url)
            em.set_author(name = ctx.author.name, icon_url = ctx.author.avatar_url)
            await ctx.send(embed = em)


    @commands.command()
    async def whois(self,ctx, user : discord.Member = None):
        if user == None:
            user = ctx.author
        em = discord.Embed(title = user.name, color = user.color)
        em.add_field(name = "ID:", value = user.id)
        em.set_thumbnail(url = user.avatar_url)
        em.set_footer(text='Bot Made by NightZan999#0194')
        em.set_author(name = ctx.author.name, icon_url = ctx.author.avatar_url)
        await ctx.send(embed = em)

    @whois.error
    async def whois_error(self, ctx, error):
        if isinstance(error, commands.BadArgument):
            em = discord.Embed(title = f"<:fail:761292267360485378> Whois Error", color = ctx.author.color)
            em.add_field(name = f"Reason:", value = f"Arguments were of the wrong data type!")
            em.add_field(name = "Args", value = "```\nimp whois [@user]\n```")
            em.set_thumbnail(url = ctx.author.avatar_url)
            em.set_author(name = ctx.author.name, icon_url = ctx.author.avatar_url)
            await ctx.send(embed = em)
            
    @commands.command(aliases = ["bi"])
    async def botinfo(self, ctx):
        embed = discord.Embed(title = "Botinfo", color = ctx.author.color,
        description = "TheImperialGod, is an awesome customizable discord bot with awesome features. Check some information about the bot below!"
        )
        embed.add_field(name = "First went live on:", value = "1 / 10 / 2020")
        embed.add_field(name = "Started coding on:", value = "26 / 9 / 2020")
        embed.add_field(name = f"Creator", value = f"NightZan999#0194")
        embed.add_field(name = 'Hosting', value = f"Chaotic Destiny Hosting ")
        embed.add_field(name = "Servers:", value = f'`{len(self.client.guilds)}`')
        embed.add_field(name = 'Customizable Settings:', value = f"Automoderation and utilities! ")
        embed.add_field(name = "Database:", value = "SQLite3")
        embed.add_field(name = "Website:", value = "<:VERIFIED_DEVELOPER:761297621502656512> [Web Dashboard](https://theimperialgod.ml)")
        embed.add_field(name = "Number of Commands:", value = f"`{len(self.client.commands)}` (including special owner commands)")
        embed.add_field(name = "**Tech:**", value = "```diff\n+ Library : discord.py\n+ Database : AIOSQLite\n+ Hosting Services : Chaotic Destiny Hosting!\n```", inline = False)
        embed.add_field(name = "Users:", value = f'`{len(self.client.users)}`')
        embed.set_footer(text='Bot Made by NightZan999#0194', icon_url = ctx.author.avatar_url)
        embed.set_author(name = ctx.author.name, icon_url = ctx.author.avatar_url)
        await ctx.send(embed = embed)

    @commands.command()
    async def ping(self, ctx):
        embed = discord.Embed(title = ":ping_pong: Pong!", color = ctx.author.color,
        description = "The number rlly doesn't matter. Smh!")
        embed.add_field(name=  "Client Latency", value = f"`{round(self.client.latency * 1000)}ms`")
        embed.set_author(name = ctx.author.name, icon_url = ctx.author.avatar_url)
        embed.set_footer(text='Bot Made by NightZan999#0194')
        await ctx.send(embed = embed)
    
    @commands.command()
    async def credits(self, ctx):
        em = discord.Embed(title = ":scroll: Credits of TheImperialGod", color = ctx.author.color, description = "Github link is [here](https://github.com/NightZan999/TheImperialGod)")
        em.add_field(name = "#1 NightZan999", value = f"""I have done everything on TheImperialGod, coded the entire bot, taken feedback, grown it to {len(self.client.guilds)} servers.\nI am even writing this right now!\nMy hopes are to you, if you like this bot type: `imp support`. That shows you ways to support TheImperialGod"\n\nI have written 70,000 lines of code for the bot and the website, so yeah-""")
        em.add_field(name = '#2 Github', value = "I did do all the coding, but I made TheImperialGod open source, this is why many people respond to my issues. Some people have corrected some glitches, and a full credits list is avalible on github")
        em.set_author(name = ctx.author.name, icon_url = ctx.author.avatar_url)
        em.set_footer(text = "invite me now!")
        await ctx.send(embed = em)
    
    @commands.command()
    async def uptime(self, ctx):
        current_time = datetime.datetime.utcnow()
        uptime = (current_time - self.launched_at)
        em = discord.Embed(title = "<:zancool:819065864153595945> My Uptime", color = ctx.author.color)
        em.add_field(name = "Uptime", value = f"I have been online for **{humanize.naturaldelta(uptime)}**")
        em.set_author(name = ctx.author.name, icon_url = ctx.author.avatar_url)
        em.set_footer(text = "Requested by {}".format(ctx.author.name), icon_url = ctx.author.avatar_url)
        await ctx.send(embed = em)
    
    @commands.command()
    async def roleinfo(self, ctx, *, role_: discord.Role = None):
        role = role_
        if role is None:
            await ctx.send("Please provide a valid role")
        em = discord.Embed(title = f"Info about {role.name}", color = ctx.author.color, description = f"Here is an insight into {role.mention}")
        em.add_field(name = "ID:", value = f"`{role.id}`")
        em.add_field(name = "Name:", value = f"`{role.name}`")
        em.add_field(name = "Server it belongs to:", value = f"{role.guild.name}", inline = True)

        em.add_field(name = "Hoisted:", value = f"`{role.hoist}`")
        em.add_field(name = "Managed by extension:", value = f"`{role.managed}`", inline = True)
        em.add_field(name = "Boost Role:", value = f"`{role.is_premium_subscriber()}`", inline = True)

        em.add_field(name = "Mentionable:", value = f"`{role.mentionable}`" )
        em.add_field(name = "Is Default:", value = f"`{role.is_default()}`", inline = True)
        em.add_field(name = "Bot Role:", value = f"`{role.is_bot_managed()}`", inline = True)

        em.add_field(name = "Color:", value = f"{role.color}")
        em.add_field(name = "Created At:", value = f"{role.created_at}", inline = True)
        em.add_field(name = "People with it:", value =f"{len(role.members)}", inline = True)
        msg = "```diff\n"
        if role.permissions.administrator:
            msg += "+ Administrator\n"
        else:
            msg += "- Administrator\n"
        if role.permissions.manage_guild:
            msg += "+ Manage Server\n"
        else:
            msg += "- Manage Server\n"
        if role.permissions.mention_everyone:
            msg += "+ Ping Everyone\n"
        else:
            msg += "- Ping Everyone\n"
        if role.permissions.manage_roles:
            msg += "+ Manage Roles\n"
        else:
            msg += "- Manage Roles\n"
        if role.permissions.manage_channels:
            msg += "+ Manage Channels\n"
        else:
            msg += "- Manage Channels\n"
        if role.permissions.ban_members:
            msg += "+ Ban Members\n"
        else:
            msg += "- Ban Members\n"
        if role.permissions.kick_members:
            msg += "+ Kick Members\n"
        else:
            msg += "- Kick Members\n"
        if role.permissions.view_audit_log:
            msg += "+ View Audit Log\n"
        else:
            msg += "- View Audit Log\n"
        if role.permissions.manage_messages:
            msg += "+ Manage Messages\n"
        else:
            msg += "- Manage Messages\n"
        if role.permissions.add_reactions:
            msg += "+ Add Reactions\n"
        else:
            msg += "- Add Reactions\n"
        if role.permissions.view_channel:
            msg += "+ Read Messages\n"
        else:
            msg += "- Read Messages\n"
        if role.permissions.send_messages:
            msg += "+ Send Messages\n"
        else:
            msg += "- Send Messages\n"
        if role.permissions.embed_links:
            msg += "+ Embed Links\n"
        else:
            msg += "- Embed Links\n"
        if role.permissions.read_message_history:
            msg += "+ Read Message History\n"
        else:
            msg += "- Read Message History\n"
        if role.permissions.view_guild_insights:
            msg += "+ View Guild Insights\n"
        else:
            msg += "- View Guild Insights\n"
        if role.permissions.connect:
            msg += "+ Join VC\n"
        else:
            msg += "- Join VC\n"
        if role.permissions.speak:
            msg += "+ Speak in VC\n"
        else:
            msg += "- Speak in VC\n"
        
        if role.permissions.change_nickname:
            msg += "+ Change Nickname\n"
        else:
            msg += "- Change Nickname\n"
        
        if role.permissions.manage_nicknames:
            msg += "+ Manage Nicknames\n"
        else:
            msg += "- Manage Nicknames\n"
        
        if role.permissions.manage_webhooks:
            msg += "+ Manage Webhooks\n"
        else:
            msg += "- Manage Webhooks\n"
        
        if role.permissions.manage_emojis:
            msg += "+ Manage Emojis\n"
        else:
            msg += "- Manage Emojis\n"
        

        msg += "\n```"
        em.add_field(name = "Permissions:", value = msg, inline = False)

        em.set_footer(text = "invite me ;)", icon_url = ctx.author.avatar_url)
        em.set_thumbnail(url = str(ctx.guild.icon_url))
        em.set_author(name = ctx.author.name, icon_url = ctx.author.avatar_url)
        await ctx.send(embed = em)

    @commands.command()
    async def evalhelp(self, ctx):
        em = discord.Embed(title = "Help with eval!", color = discord.Color.random(), description = "Check this help image for help with Jishaku!")
        em.set_image(url = "https://ibb.co/MhythwM")
        await ctx.send(embed = em)

    @commands.command()
    async def stats(self, ctx):
        em = discord.Embed(title=  "Stats about me", color = self.client.user.color, description = "My stats :partying_face:")
        em.add_field(name = "Users:", value = f"{len(self.client.users)}")
        em.add_field(name = "Servers:", value = f"{len(self.client.guilds)}")
        em.add_field(name = "Total Commands:", value = f"{len(self.client.commands)}")
        em.add_field(name = "Channels:", value = f"{len(self.client.channels)}")
        await ctx.send(embed = em)
    
def setup(client):
    client.add_cog(Information(client))