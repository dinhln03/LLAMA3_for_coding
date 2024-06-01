import argparse
import logging
from pprint import pformat

from . import guide
from . import settings


log = logging.getLogger(__name__)


def cli(settingsobject=None):
    parser = argparse.ArgumentParser(description='Create a CSS/LESS/SASS style guide.')
    if not settingsobject:
        parser.add_argument('-f', '--settingsfile',
            dest='settingsfile', default='vitalstyles.json',
            help='Path to settings file. Defaults to "vitalstyles.json".')
    parser.add_argument('-l', '--loglevel',
        dest='loglevel', default='INFO',
        choices=['DEBUG', 'INFO', 'ERROR'], help='Loglevel.')
    args = parser.parse_args()

    loglevel = getattr(logging, args.loglevel)
    logging.basicConfig(
        format='[%(name)s] %(levelname)s: %(message)s',
        level=loglevel
    )

    if loglevel > logging.DEBUG:
        markdownlogger = logging.getLogger('MARKDOWN')
        markdownlogger.setLevel(logging.WARNING)

    if not settingsobject:
        settingsobject = settings.Settings(args.settingsfile)
    logging.debug('Creating vitalstyles styleguide with the following settings:\n%s',
                  pformat(settingsobject.settings))
    guide.Guide(settingsobject).render()


if __name__ == '__main__':
    cli()
