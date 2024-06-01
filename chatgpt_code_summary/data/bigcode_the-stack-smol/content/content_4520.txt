"""Main entry point for VarFish CLI."""

import argparse
import logging
import os
import sys

import logzero
import toml
from logzero import logger

from varfish_cli import __version__
from .common import run_nocmd, CommonConfig
from .case import setup_argparse as setup_argparse_case
from .case import run as run_case

#: Paths to search the global configuration in.
GLOBAL_CONFIG_PATHS = ("~/.varfishrc.toml",)


def setup_argparse_only():  # pragma: nocover
    """Wrapper for ``setup_argparse()`` that only returns the parser.

    Only used in sphinx documentation via ``sphinx-argparse``.
    """
    return setup_argparse()[0]


def setup_argparse():
    """Create argument parser."""
    # Construct argument parser and set global options.
    parser = argparse.ArgumentParser(prog="varfish-cli")
    parser.add_argument("--verbose", action="store_true", default=False, help="Increase verbosity.")
    parser.add_argument("--version", action="version", version="%%(prog)s %s" % __version__)

    group = parser.add_argument_group("Basic Configuration")
    group.add_argument(
        "--no-verify-ssl",
        dest="verify_ssl",
        default=True,
        action="store_false",
        help="Disable HTTPS SSL verification",
    )
    group.add_argument(
        "--config",
        default=os.environ.get("VARFISH_CONFIG_PATH", None),
        help="Path to configuration file.",
    )
    group.add_argument(
        "--varfish-server-url",
        default=os.environ.get("VARFISH_SERVER_URL", None),
        help="VarFish server URL key to use, defaults to env VARFISH_SERVER_URL.",
    )
    group.add_argument(
        "--varfish-api-token",
        default=os.environ.get("VARFISH_API_TOKEN", None),
        help="VarFish API token to use, defaults to env VARFISH_API_TOKEN.",
    )

    # Add sub parsers for each argument.
    subparsers = parser.add_subparsers(dest="cmd")

    setup_argparse_case(subparsers.add_parser("case", help="Work with cases."))

    return parser, subparsers


def main(argv=None):
    """Main entry point before parsing command line arguments."""
    # Setup command line parser.
    parser, subparsers = setup_argparse()

    # Actually parse command line arguments.
    args = parser.parse_args(argv)

    # Setup logging incl. verbosity.
    if args.verbose:  # pragma: no cover
        level = logging.DEBUG
    else:
        # Remove module name and line number if not running in debug mode.s
        formatter = logzero.LogFormatter(
            fmt="%(color)s[%(levelname)1.1s %(asctime)s]%(end_color)s %(message)s"
        )
        logzero.formatter(formatter)
        level = logging.INFO
    logzero.loglevel(level=level)

    # Load configuration, if any.
    if args.config:
        config_paths = (args.config,)
    else:
        config_paths = GLOBAL_CONFIG_PATHS
    for config_path in config_paths:
        config_path = os.path.expanduser(os.path.expandvars(config_path))
        if os.path.exists(config_path):
            with open(config_path, "rt") as tomlf:
                toml_config = toml.load(tomlf)
            break
    else:
        toml_config = None
        logger.info("Could not find any of the global configuration files %s.", config_paths)

    # Merge configuration from command line/environment args and configuration file.
    config = CommonConfig.create(args, toml_config)

    # Handle the actual command line.
    cmds = {None: run_nocmd, "case": run_case}

    res = cmds[args.cmd](
        config, toml_config, args, parser, subparsers.choices[args.cmd] if args.cmd else None
    )
    if not res:
        logger.info("All done. Have a nice day!")
    else:  # pragma: nocover
        logger.error("Something did not work out correctly.")
    return res


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
