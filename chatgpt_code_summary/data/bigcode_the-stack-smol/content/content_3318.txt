"""Commands for starting daemons."""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import pprint

import confpy.api
import confpy.core.option

from .. import messages


cfg = confpy.api.Configuration(
    transport=confpy.api.Namespace(
        description='Message transport options.',
        source=confpy.core.option.Option(
            description='The transport to fetch new requests from.',
            required=True,
        ),
        error=confpy.core.option.Option(
            description='The transport to which errors are written.',
            required=True,
        ),
        result=confpy.core.option.Option(
            description='The transport to which results are written.',
            required=True,
        ),
    ),
    daemon=confpy.api.Namespace(
        description='Long running daemon options.',
        profiler=confpy.core.option.Option(
            description='The profiler implementation to use.',
            required=True,
        ),
        process=confpy.core.option.Option(
            description='The daemon interface implemention to use.',
            required=True,
        ),
        pidfile=confpy.api.StringOption(
            description='The location to use as a pidfile.',
            required=True,
        ),
    ),
)


def _common_args():
    """ArgumentParser setup for all CLI commands."""
    parser = argparse.ArgumentParser(
        description='Start a new profiler process.'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='The Python configuration file for the process.',
    )
    return parser


def profiler_main():
    """Manage a profiler daemon."""
    parser = _common_args()
    parser.add_argument(
        '--action',
        required=True,
        choices=('start', 'stop', 'restart'),
    )

    args, _ = parser.parse_known_args()
    cfg = confpy.api.parse_options(files=(args.config,), env_prefix='PYPERF')

    proc = cfg.daemon.process(
        source_transport=cfg.transport.source,
        error_transport=cfg.transport.error,
        results_transport=cfg.transport.result,
        profiler=cfg.daemon.profiler,
        pidfile=cfg.daemon.pidfile,
    )

    if args.action == 'stop':

        proc.stop()

    if args.action == 'start':

        proc.start()

    if args.action == 'restart':

        proc.restart()


def send_request():
    """Send a profile request to the daemon."""
    parser = _common_args()
    parser.add_argument(
        '--identifier',
        required=True,
        help='The unique message identifier.',
    )
    parser.add_argument(
        '--setup',
        default='pass',
        help='Any setup code if needed for the profile.',
    )
    parser.add_argument(
        '--code',
        required=True,
        help='The code to profile.',
    )

    args, _ = parser.parse_known_args()
    cfg = confpy.api.parse_options(files=(args.config,), env_prefix='PYPERF')

    cfg.transport.source().send(
        messages.ProfileRequest(
            identifier=args.identifier,
            setup=args.setup,
            code=args.code,
        ),
    )


def fetch_result():
    """Fetch a result from the transport."""
    parser = _common_args()
    args, _ = parser.parse_known_args()
    cfg = confpy.api.parse_options(files=(args.config,), env_prefix='PYPERF')

    transport = cfg.transport.result()
    msg = transport.fetch()
    if msg is not None:

        transport.complete(msg)
        pprint.pprint(msg.json)


def fetch_error():
    """Fetch an error from the transport."""
    parser = _common_args()
    args, _ = parser.parse_known_args()
    cfg = confpy.api.parse_options(files=(args.config,), env_prefix='PYPERF')

    transport = cfg.transport.error()
    msg = transport.fetch()
    if msg is not None:

        transport.complete(msg)
        pprint.pprint(msg.json)
