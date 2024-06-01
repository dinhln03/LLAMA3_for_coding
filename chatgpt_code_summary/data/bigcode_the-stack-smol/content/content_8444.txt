import discord
from discord.ext import commands


class AttackStrats(commands.Cog):

    def __init__(self, bot):
        self.bot = bot




    @commands.group()
    async def strat(self, ctx):
        if ctx.invoked_subcommand is None:
            await ctx.send("You need to specify a townhall and strategy! Available townhalls:\nth8\nth9\nth10\nth11\nth12\nth13")



    @strat.group()
    async def th8(self, ctx):
        if ctx.invoked_subcommand is None:
            await ctx.send("You need to specify an attack strategy!Available strategies:\nhogs\ndragons")


    @th8.command(name="hogs")
    async def th8_hogs(self, ctx):

        th8_hogsEmbed = discord.Embed(
            title="TH8 Hogs", 
            description="**ARMY COMPOSITION:**```\nTroops: 32 Hog Riders, 10 Wizards\nSpells: 3 Heals, 1 Poison\nCC: 5 high level hog riders\n```\n **ATTACKING:\n**", 
            color=self.bot.colors
        )

        th8_hogsEmbed.add_field(
            name="Step 1- Dealing the clan castle troops:", 
            value="Send a hog in to lure out the cc. If it doesn't come out the send another one. Luring out the cc is very important because it can destroy all of your hogs. Once the cc is lured, drop your poison on the cc, drop your king and about 5 wizards. The king will suck up the damage while the wizards take out the cc.\n", 
            inline=False
        )

        th8_hogsEmbed.add_field(
            name="Step 2- Attacking:", 
            value="Drop your hogs and cc on the defenses once the enemy cc is gone. It is better to drop them on each defense, but don't spread them out too much or they wont be effective. As soon as the hogs have taken out the first initial defense drop your wizards behind. Clean up is very important with hogs. As your hogs make their way around the base, they have to be kept healed. Drop heal spells as the hogs go while looking out for giant bombs. Giant bombs will take out all of your hogs, so make sure you are watching and are ready to drop a spell when you see one.\n", 
            inline=False
        )

        th8_hogsEmbed.add_field(
            name="Step 3- Clean up:", 
            value="Once all the defenses are destroyed the wizards should have cleaned up a lot, and the hogs will then take care of the last few buildings.\n", 
            inline=False
        )


        await ctx.send(embed=th8_hogsEmbed)




    @th8.command(name="dragons")
    async def th8_dragons(self, ctx):

        th8_dragonsEmbed = discord.Embed(
            title="TH8 Dragons",
            description="**ARMY COMPOSITION**```\nTroops: 10 Dragons\nSpells: 3 Rages, 1 Poison\nCC: Balloons\n```\n**ATTACKING**\n",
            color = self.bot.colors
        )

        th8_dragonsEmbed.add_field(
            name="Step 1- Funneling:",
            value="Drop your king on one side of the base and a dragon on the other side. We want our dragons to go to the center, not go around the base. The king and dragon will funnel so our main army goes down to the middle.",
            inline=False
        )

        th8_dragonsEmbed.add_field(
            name="Step 2- Main Army:",
            value=""
        )





        await ctx.send(embed=th8_dragonsEmbed)













    @strat.group()
    async def th9(self, ctx):
        if ctx.invoked_subcommand is None:
            await ctx.send("You need to specify an attack strategy!Available strategies:\n")


    # @th9.command(name="dragons")
    # async def dragons(self, ctx):
    #     await ctx.send("")














    @strat.group()
    async def th10(self, ctx):
        if ctx.invoked_subcommand is None:
            await ctx.send("You need to specify an attack strategy!Available strategies:\n")
















    @strat.group()
    async def th11(self, ctx):
        if ctx.invoked_subcommand is None:
            await ctx.send("You need to specify an attack strategy!Available strategies:\nhybrid")



    @th11.command(name="hybrid")
    async def th11_qc_hybrid(self, ctx):
        th11_qc_hybridEmbed = discord.Embed(
            title="TH11 Queen Charge Hybrid", 
            description="**ARMY COMPOSITION:**```\nTroops: 5 Healers (for queen charge), 1 or 2 Balloons (to protect the healers), 2 Baby Dragons (for funnel), 15 or 16 Miners, 10 Hogs, 1 or 2 Super Wall Breakers (if you don't have them replace with regular wall breakers), cleanup troops with the remaining space (archers or minions)\nSpells: 2 Heals, 2 Rages, 2 Freeze, 1 Poison\nCC: More Hogs and a Rage or Heal (whatever you think you need), Siege Barracks\n```\n **ATTACKING:\n**", 
            color=discord.Color.dark_gold())

        th11_qc_hybridEmbed.add_field(
            name="Step 1- Queen Charge:", 
            value="Identify where to charge your queen into the base. Remember the purpose of the queen walk is to get rid of clan castle troops. Drop the baby dragons to funnel for the Queen to make sure she goes into the base. Then drop your healers and a loon or two to look for black bombs (seeking air mines that will take out a healer). Wall break into the compartment you need your queen to go in. When your queen comes under heavy fire make sure to drop the rage on both the queen and healers. If your queen is under target by a single inferno make sure to freeze it. Once you get the cc pull make sure to POISON them. You can take care of an edrag super easily if you have a poison.\n", 
            inline=False)

        th11_qc_hybridEmbed.add_field(
            name="Step 2- Funneling:",
            value="Once we have dealt with the clan castle we need to identify where we're going to be sending in the Hybrid portion of the attack. Normally the Queen Charge will have taken out a chunk of the base on one of the sides. We want our hybrid to go straight for the core of the base and not down the sides. Think of the QC as one side of a funnel for your Hybrid. For the other side of the funnel we need to place our King and Siege Barracks down the other edge of the base to clear all the trash buildings (collectors, barracks, etc.) so our hybrid has a clear path into the core of the base.\n", 
            inline=False
        )

        th11_qc_hybridEmbed.add_field(
            name="Step 3- The Main Army:",
            value="Once the King and Siege Barracks have cleared quite a bit, we want to start our hybrid by placing all Miners, Hogs and Warden down to go on the path into the core of the base. If you didn't take out the Eagle with your Queen Charge use the Warden ability to protect the hybrid from the strike. Freezes can also be used here to freeze up Multi Infernos or a section with a lot of splash damage like Wizard Towers, Mortars or Bomb Towers. Place Heals where necessary to keep the hybrid alive.\n"
        )

        th11_qc_hybridEmbed.add_field(
            name="Step 4- Cleanup:",
            value="We also have to think about placing cleanup troops for corner builder huts, missed buildings on the other side of the base etc.\n", 
            inline=False
        )

        await ctx.send(embed=th11_qc_hybridEmbed)













    @strat.group()
    async def th12(self, ctx):
        if ctx.invoked_subcommand is None:
            await ctx.send("You need to specify an attack strategy!Available strategies:\n")


















    @strat.group()
    async def th13(self, ctx):
        if ctx.invoked_subcommand is None:
            await ctx.send("You need to specify an attack strategy!Available strategies:\n")





























def setup(bot):
    bot.add_cog(AttackStrats(bot))
