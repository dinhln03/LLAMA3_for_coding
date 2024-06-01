from os import environ

from scrapy_heroku.poller import Psycopg2QueuePoller
from scrapy_heroku.scheduler import Psycopg2SpiderScheduler
from scrapyd.eggstorage import FilesystemEggStorage
from scrapyd.environ import Environment
from scrapyd.interfaces import (IEggStorage, IEnvironment, IPoller,
                                ISpiderScheduler)
from scrapyd.launcher import Launcher
from scrapyd.website import Root
from twisted.application.internet import TCPServer, TimerService
from twisted.application.service import Application
from twisted.python import log
from twisted.web import server


def application(config):
    app = Application("Scrapyd")
    http_port = int(environ.get('PORT', config.getint('http_port', 6800)))
    config.cp.set('scrapyd', 'database_url', environ.get('DATABASE_URL'))

    poller = Psycopg2QueuePoller(config)
    eggstorage = FilesystemEggStorage(config)
    scheduler = Psycopg2SpiderScheduler(config)
    environment = Environment(config)

    app.setComponent(IPoller, poller)
    app.setComponent(IEggStorage, eggstorage)
    app.setComponent(ISpiderScheduler, scheduler)
    app.setComponent(IEnvironment, environment)

    launcher = Launcher(config, app)
    timer = TimerService(5, poller.poll)
    webservice = TCPServer(http_port, server.Site(Root(config, app)))
    log.msg("Scrapyd web console available at http://localhost:%s/ (HEROKU)" %
            http_port)

    launcher.setServiceParent(app)
    timer.setServiceParent(app)
    webservice.setServiceParent(app)

    return app
