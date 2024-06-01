import logging.config

from netdisco.discovery import NetworkDiscovery

LOG = logging.getLogger(__name__)


def discover():
    hue_bridges = []

    LOG.info('Searching for Hue devices...')

    netdis = NetworkDiscovery()
    netdis.scan()

    for dev in netdis.discover():
        for info in netdis.get_info(dev):
            if 'name' in info and 'Philips hue' in info['name']:
                hue_bridges.append(info)
                LOG.info('Hue bridge found: %s', info['host'])

    netdis.stop()

    if len(hue_bridges) == 1:
        return hue_bridges[0]['host']

    if len(hue_bridges) == 2:
        LOG.warning('More than one Hue bridge found.')
    elif not hue_bridges:
        LOG.warning('No Hue bridges found.')

    return None
