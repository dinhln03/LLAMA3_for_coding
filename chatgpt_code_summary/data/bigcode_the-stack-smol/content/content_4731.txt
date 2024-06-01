import logging
from logging import basicConfig, getLogger

from lib import do_something


def main():
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    basicConfig(format=log_fmt,
                level="DEBUG")
    logger = getLogger(__name__)

    for i in range(3):
        logger.info("---------------{}-th trial------------".format(i))
        logger.debug("debugging...")
        logger.info("tell information")
        logger.warn("warn it comes from something like ...")
        logger.error("Ops some error is occured")
        logger.critical("critical event happened")
        logger.debug("It's raining again")
        logger.info("with hail the size of hailstones")
        do_something()


if __name__ == '__main__':
    main()
