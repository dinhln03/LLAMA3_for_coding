# from imports import *
# import random

# class Docs(commands.Cog):
#   def __init__(self, bot):
#     self.bot = bot

#     self.bot.loop.create_task(self.__ainit__())

#   async def __ainit__(self):
#     await self.bot.wait_until_ready()
#     self.scraper = AsyncScraper(session = self.bot.session)

#   async def rtfm_lookup(self, program = None, *, args = None):

#     rtfm_dictionary = {
# 		"fusion.py": "https://fusion.senarc.org/en/master/",
#         "development" : "https://fusion.senarc.org/en/development/"
#     }

#     if not args:
#       return rtfm_dictionary.get(program)

#     else:
#       url = rtfm_dictionary.get(program)

#       results = await self.scraper.search(args, page=url)

#       if not results:
#         return f"Could not find anything with {args}."

#       else:
#         return results

#   def reference(self, message):
#     reference = message.reference

#     if reference and isinstance(reference.resolved, discord.Message):
# 	    return reference.resolved.to_reference()

#     return None

#   async def rtfm_send(self, ctx, results):

#     if isinstance(results, str):
#       await ctx.send(results, allowed_mentions = discord.AllowedMentions.none())

#     else:
#       embed = discord.Embed(color = random.randint(0, 16777215))

#       results = results[:10]
#       embed.description = "\n".join(f"[`{result}`]({value})" for result, value in results)

#       reference = self.reference(ctx.message)
#       await ctx.send(embed=embed, reference = reference)

#   @commands.group(slash_interaction=True, aliases=["rtd", "rtfs"], brief="Search for attributes from docs.")
#   async def rtfm(self, ctx, *, args = None):

#     await ctx.trigger_typing()
#     results = await self.rtfm_lookup(program = "fusion.py", args = args)
#     await self.rtfm_send(ctx, results)

#   @rtfm.command(slash_interaction=True, brief = "a command using doc_search to look up at development's docs")
#   async def development(self, ctx, *, args = None):

#     await ctx.trigger_typing()
#     results = await self.rtfm_lookup(program="development", args = args)
#     await self.rtfm_send(ctx, results)

# def setup(bot):
# 	bot.add_cog(Docs(bot))
