"""
Support for Syslog-based networking devices.
For now, support is limited to hostapd and dnsmasq.

Example syslog lines:

    <30>Dec 31 13:03:21 router hostapd: wlan1: STA a4:77:33:e3:17:7c WPA: group key handshake completed (RSN)
    <29>Dec 31 13:05:15 router hostapd: wlan0: AP-STA-CONNECTED 64:20:0c:37:52:82
    <30>Dec 31 13:15:22 router hostapd: wlan0: STA 64:20:0c:37:52:82 IEEE 802.11: disassociated
    <30>Dec 31 13:15:23 router hostapd: wlan0: STA 64:20:0c:37:52:82 IEEE 802.11: deauthenticated due to inactivity (timer DEAUTH/REMOVE)
    <29>Dec 31 13:20:15 router hostapd: wlan0: AP-STA-CONNECTED 64:20:0c:37:52:82
    <30>Dec 31 13:02:33 router dnsmasq-dhcp[1601]: DHCPACK(br-lan) 192.168.0.101 f4:6d:04:ae:ac:d7 leon-pc
"""

from asyncio import coroutine
from collections import namedtuple
from datetime import timedelta
import logging
import voluptuous as vol

from homeassistant.components.device_tracker import PLATFORM_SCHEMA, SOURCE_TYPE_ROUTER
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_DEVICES
from homeassistant.helpers.event import async_track_time_interval
import homeassistant.helpers.config_validation as cv

_LOGGER = logging.getLogger(__name__)


PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend({
    #vol.Optional(CONF_WHITELIST): cv.string,  # ACL
    vol.Required(CONF_HOST): cv.string,
    vol.Optional(CONF_PORT, default=514): cv.port,

    # mac => name
    vol.Required(CONF_DEVICES): {cv.string: cv.string},

    # TODO: TCP vs UDP
    # TODO: periodically ARP ping wired devices
})

Event = namedtuple('Event', 'mac kind is_sta reason')


STA_EVENTS = {
    'WPA: group key handshake completed': 'home',
    'WPA: pairwise key handshake completed': 'home',

    'deauthenticated due to local deauth request': 'not_home',

    'IEEE 802.11: disconnected due to excessive missing ACKs': 'timeout',
    'IEEE 802.11: disassociated due to inactivity': 'timeout',
    'IEEE 802.11: deauthenticated due to inactivity': 'timeout',

    # Ignored, should be covered by AP-STA-*
    'IEEE 802.11: associated': '',
    'IEEE 802.11: authenticated': '',
    'IEEE 802.11: disassociated': '',
}


def _skip_date_tokens(tokens):
    """
    Based on RFC 3164 + RFC 5424 and real-world logs
    """
    if tokens and tokens[0].startswith('<'):
        tokens.pop(0)
    while tokens and (not tokens[0] or tokens[0][:1].isdigit()):
        tokens.pop(0)


def _find_process(tokens):
    while tokens:
        token = tokens.pop(0)
        if token.endswith(':'):
            c = token.find('[')
            if c > -1:
                return token[:c]
            return token[:-1]

def _remove_param(tokens):
    i = len(tokens) - 1
    while i > 0:
        if tokens[i].startswith('('):
            return tokens[:i]
        i -= 1
    return tokens


def parse_syslog_line(line):
    """Parses lines created by hostapd and dnsmasq DHCP"""
    tokens = line.split(' ')
    _skip_date_tokens(tokens)
    process = _find_process(tokens)
    if not process or not tokens:
        _LOGGER.debug('Unable to process line: %r', line)
        return
    if process == 'hostapd':
        # <iface>: AP-STA-<event>: <mac>
        if len(tokens) == 3:
            if tokens[1] == 'AP-STA-CONNECTED':
                return Event(tokens[2], 'home', True, tokens[1])
            elif tokens[1] == 'AP-STA-DISCONNECTED':
                # Disconnected, but we might get the real reason later
                return Event(tokens[2], 'timeout', True, tokens[1])
        elif len(tokens) > 4 and tokens[1] == 'STA':
            # <iface>: STA <mac> WPA: <...>
            # <iface>: STA <mac> IEEE 802.11: <...>
            suffix = ' '.join(_remove_param(tokens[3:]))
            for consider, status in STA_EVENTS.items():
                if suffix.endswith(consider):
                    if status == '':
                        return
                    return Event(tokens[2], status, True, suffix)
            _LOGGER.warning('Unhandled line: %r', line)
    elif process == 'dnsmasq-dhcp':
        if len(tokens) >= 3:
            # <event>(<iface> <ip> <mac> <name>
            if tokens[0].startswith('DHCPACK('):
                return Event(tokens[2], 'home', False, tokens[0])


class SyslogScanner:
    def __init__(self, hass, async_see, devices):
        self.hass = hass
        self.devices = devices
        self.wireless_devices = set()
        self.async_see = async_see
        # TODO: consider marking all devices as offline after start
        self.debug_marked = {}
        #async_track_time_interval(hass, self.scan_online_devices,
        #                          timedelta(minutes=1))

    @coroutine
    def scan_online_devices(self, now=None):
        _LOGGER.info('Check online devices')
        for mac, name in self.devices.items():
            if mac in self.wireless_devices:
                continue
            _LOGGER.info('Check %r', mac)

    def process_line(self, line):
        event = parse_syslog_line(line.rstrip('\n'))
        if not event:
            return
        _LOGGER.info('%r', event)

        mac = event.mac.replace(':', '')
        if event.is_sta:
            self.wireless_devices.add(mac)

        device = self.devices.get(mac)
        if not device:
            # Automatic tracking
            device = self.devices[mac] = mac

        consider_home = None
        state = event.kind
        if event.kind == 'timeout':
            state = 'not_home'
            # TODO: this feature has not been added yet
            consider_home = timedelta(minutes=5)

        if self.debug_marked.get(device) != state:
            _LOGGER.info('Mark %r as %r [%s]', device, state, consider_home)
            self.debug_marked[device] = state

        self.hass.async_add_job(self.async_see(dev_id=device,
                                               source_type=SOURCE_TYPE_ROUTER,
                                               mac=event.mac,
                                               #consider_home=consider_home,
                                               location_name=state))


class SyslogScannerUDP(SyslogScanner):
    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        message = data.decode('utf8', 'replace')
        self.process_line(message)


@coroutine
def async_setup_scanner(hass, config, async_see, discovery_info=None):
    bind = (config[CONF_HOST], config[CONF_PORT])
    _LOGGER.info('Listening on %s:%s', bind[0], bind[1])
    proto = lambda: SyslogScannerUDP(hass, async_see, config[CONF_DEVICES])
    listen = hass.loop.create_datagram_endpoint(proto, local_addr=bind)
    hass.async_add_job(listen)
    return True
