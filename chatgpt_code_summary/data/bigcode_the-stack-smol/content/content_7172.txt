import argparse
import logging
from pprint import pprint

from .gcal import GCal

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TODO"
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
    )
    return parser.parse_args()


def setup_logging(args: argparse.Namespace) -> None:
    level = (
        logging.DEBUG if args.verbose >= 2 else
        logging.INFO if args.verbose >= 1 else
        logging.WARNING
    )
    logging.basicConfig(level=level)


def main() -> None:
    args = parse_args()
    setup_logging(args)

    cal = GCal()
    pprint(list(cal.list_calendars(fields="items(id,summary)")))
