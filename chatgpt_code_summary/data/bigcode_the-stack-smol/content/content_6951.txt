import ref_bot.cog.articlerefs
import importlib
import conf

from discord.ext import commands
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def setup_dbsession():
    engine = create_engine(conf.ini_config.get('sqlalchemy', 'connection_string'))    
    
    sessionm = sessionmaker()
    sessionm.configure(bind=engine)
    return sessionm()

def setup(bot):
    print('ref_bot extension loading.')

    dbsession = setup_dbsession()
    importlib.reload(ref_bot.cog.articlerefs)
    bot.remove_command('help')
    bot.add_cog(ref_bot.cog.articlerefs.ArticleRefs(bot, dbsession))


def teardown(bot):
    print('ref_bot extension unloading')
    bot.remove_cog('ArticleRefs')
