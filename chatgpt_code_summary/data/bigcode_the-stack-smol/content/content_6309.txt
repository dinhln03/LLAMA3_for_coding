import logging

from redlib.api.misc import Logger

log = Logger(name='jekt')
log.start('stdout', logging.DEBUG)

