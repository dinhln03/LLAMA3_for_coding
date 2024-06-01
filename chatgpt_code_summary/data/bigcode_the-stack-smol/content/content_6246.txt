#!/usr/bin/env python3
"""
Convert data received from alfred (ffbi format) and serve them as prometheus python client

Typical call::

    alfred -r 69 -u /var/run/alfred/alfred.sock > robin.txt
    ./robin_prometheus.py  -m robin.txt

Dependencies:
    prometheus_client -> pip3 install prometheus_client
License: CC BY 4.0
Author: Jonas Hess
Strongly Inspired by map-backend of Moritz Warning and Julian Rueth
"""

import sys
import zlib
import re
import datetime
import os
import pickle
import time
import json
import jsonschema

from prometheus_client import start_http_server
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily, REGISTRY

if sys.version_info[0] < 3:
    raise Exception("map-backend.py must be executed with Python 3.")

NOW_TIMESTAMP = datetime.datetime.utcnow().replace(microsecond=0)


class CustomCollector:
    """
    Data Collector for serving them in prometheus client
    """

    def collect(self):
        """
        collectors only function called collect. and it collects data
        """

        downstream = GaugeMetricFamily('node_bw_wan_bps', 'last tested wan downstream mb/s', labels=['nodeid'])
        for node in GLOBAL_NODES['nodes']:
            if 'downstream_mbps_wan' in node:
                downstream.add_metric([node['id']], node['downstream_mbps_wan'])
        yield downstream

        upstream = GaugeMetricFamily('node_bw_ff_bps', 'last tested ff downstream in mb/s', labels=['nodeid'])
        for node in GLOBAL_NODES['nodes']:
            if 'downstream_mbps_ff' in node:
                upstream.add_metric([node['id']], node['downstream_mbps_ff'])
        yield upstream

        ping = GaugeMetricFamily('node_gw_ping_ms', 'last tested gateway ping in ms', labels=['nodeid'])
        for node in GLOBAL_NODES['nodes']:
            if 'gw_ping_ms' in node:
                ping.add_metric([node['id']], node['gw_ping_ms'])
        yield ping

        # 'test_host': self.properties['test_host'],
        # 'tested_when': self.properties['tested_when'],

        rx_counter = CounterMetricFamily('node_rx_bytes', 'received bytes', labels=['nodeid'])
        for node in GLOBAL_NODES['nodes']:
            if 'rx_bytes' in node:
                rx_counter.add_metric([node['id']], int(node['rx_bytes']))
        yield rx_counter

        tx_counter = CounterMetricFamily('node_tx_bytes', 'transmitted bytes', labels=['nodeid'])
        for node in GLOBAL_NODES['nodes']:
            if 'tx_bytes' in node:
                tx_counter.add_metric([node['id']], int(node['tx_bytes']))
        yield tx_counter


class AlfredParser:
    """
    A class providing static methods to parse and validate data reported by
    nodes via alfred.
    """
    MAC_RE = "^([0-9a-f]{2}:){5}[0-9a-f]{2}$"
    MAC_SCHEMA = {"type": "string", "pattern": MAC_RE}
    ALFRED_NODE_SCHEMA = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            'downstream_mbps_wan': {"type": "number"},
            'downstream_mbps_ff': {"type": "number"},
            'gw_ping_ms': {"type": "number"},
            'tested_when': {"type": "string", "maxLength": 50},
            'rx_bytes': {"type": "number"},
            'tx_bytes': {"type": "number"},
        },
        "definitions": {
            "MAC": MAC_SCHEMA,
        }
    }

    @staticmethod
    def _parse_string(parse_it):
        """
        Strip an escaped string which is enclosed in double quotes and
        unescape.
        """
        if parse_it[0] != '"' or parse_it[-1] != '"':
            raise ValueError("malformatted string: {0:r}".format(parse_it))
        return bytes(parse_it[1:-1], 'ascii').decode('unicode-escape')

    @staticmethod
    def parse_line(item, nodes=None):
        """
        Parse and validate a line as returned by alfred.

        Such lines consist of a nodes MAC address and an escaped string of JSON
        encoded data. Note that most missing fields are populated with
        reasonable defaults.
        """

        # parse the strange output produced by alfred { MAC, JSON },
        if nodes is None:
            nodes = {}
        if item[-2:] != "}," or item[0] != "{":
            raise ValueError("malformatted line: {0}".format(item))
        mac, properties = item[1:-2].split(',', 1)

        # the first part must be a valid MAC
        mac = AlfredParser._parse_string(mac.strip())
        jsonschema.validate(mac, AlfredParser.MAC_SCHEMA)

        # the second part must conform to ALFRED_NODE_SCHEMA
        properties = AlfredParser._parse_string(properties.strip())

        if "\x00" in properties:
            decompress = zlib.decompressobj(zlib.MAX_WBITS | 32)
            # ignores any output beyond 64k (protection from zip bombs)
            properties = decompress.decompress(properties.encode('raw-unicode-escape'), 64 * 1024).decode('utf-8')
        else:
            properties = properties.encode('latin-1').decode('utf8')

        properties = json.loads(properties)
        jsonschema.validate(properties, AlfredParser.ALFRED_NODE_SCHEMA)

        # set some defaults for unspecified fields
        properties.setdefault('downstream_mbps_wan', 0)
        properties.setdefault('downstream_mbps_ff', 0)
        properties.setdefault('rx_bytes', 0)
        properties.setdefault('tx_bytes', 0)

        if mac in nodes:
            # update existing node
            node = nodes[mac]
            node.update_properties(properties, True)
            node.online = True
            node.lastseen = NOW_TIMESTAMP
        else:
            # create a new Node
            node = Node(mac, properties, True)
            nodes[mac] = node


class Node:
    """
    A node in the freifunk network, identified by its primary MAC.
    """

    def __init__(self, mac, properties, online):
        self.mac = mac
        self.properties = properties

        if online:
            self.lastseen = NOW_TIMESTAMP
            self.firstseen = NOW_TIMESTAMP
        else:
            self.lastseen = None
            self.firstseen = None

        self.online = online
        self.index = None  # the index of this node in the list produced for ffmap
        self.done = False

    def update_properties(self, properties, force=True):
        """
        Replace any properties with their respective values in ``properties``.
        """
        if force:
            # discard all previous properties
            self.properties = dict(properties)
            if 'force' in self.properties:
                del self.properties['force']
        else:
            # add new key/value pairs only if not already set
            for key, value in properties.items():
                if key not in self.properties:
                    if key == "force":
                        continue

                    if key == "name":
                        value = value + "*"

                    self.properties[key] = value

    def nodelist(self):
        """
        define/load the nodelist and the properties each single node has
        """

        if 'downstream_mbps_wan' not in self.properties:
            self.properties['downstream_mbps_wan'] = 0

        if 'downstream_mbps_ff' not in self.properties:
            self.properties['downstream_mbps_ff'] = 0

        obj = {
            'id': re.sub('[:]', '', self.mac),
            'status': {
                'online': self.online,
            },

            'downstream_mbps_wan': self.properties['downstream_mbps_wan'],
            'downstream_mbps_ff': self.properties['downstream_mbps_ff'],
            'tested_when': self.properties['tested_when'],
            'rx_bytes': self.properties['rx_bytes'],
            'tx_bytes': self.properties['tx_bytes'],
        }

        if 'gw_ping_ms' in self.properties:
            obj['gw_ping_ms'] = self.properties['gw_ping_ms']

        if self.firstseen:
            obj['firstseen'] = self.firstseen.isoformat()

        if self.lastseen:
            obj['status']['lastcontact'] = self.lastseen.isoformat()

        return obj


def render_nodelist(nodes):
    """
    render a nodelist out of all nodes found
    """
    all_nodes = []

    for node in nodes.values():
        all_nodes.append(node.nodelist())

    return {
        "version": "1.0.0",
        "updated_at": NOW_TIMESTAMP.isoformat(),
        'nodes': all_nodes,
    }


def load_nodes(path):
    """
    load nodes from storage file
    """
    nodes = {}
    with open(path, 'rb') as file:
        nodes = pickle.load(file)

    for node in nodes.values():
        # reset old properties
        node.online = False
        node.index = None
        node.clientcount = 0

    return nodes


def save_nodes(path, nodes):
    """
    save nodes to storage file
    """
    with open(path, 'wb') as file:
        pickle.dump(nodes, file)


def remove_old_nodes(nodes, delta):
    """
    remove nodes older than a certain limit
    """
    limit = NOW_TIMESTAMP - delta
    old_keys = []

    for key, node in nodes.items():
        if node.lastseen < limit:
            old_keys.append(key)

    count = 0
    for key in old_keys:
        del nodes[key]
        count += 1

    print("Removed {} old nodes".format(count))


def is_file(path):
    """
    just check whether there is a file on given path
    """
    return path and os.path.isfile(path)


def main():
    """
    main function collecting data from input file/storage and serving prometheus data
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert data received from alfred and provide them as prometheus-service')
    parser.add_argument('-m', '--maps', default='robin.txt', help=r'input file containing data collected by alfred')
    parser.add_argument('--storage', default='nodes_backup.bin',
                        help=r'store old data between calls e.g. to remember node lastseen values')
    parser.add_argument('-p', '--port', default=8000, help=r'the port this service should listen to')

    args = parser.parse_args()

    # mac => node
    nodes = {}

    # load old nodes that we have stored from the last call of this script,
    # that way we can show nodes that are offline
    if is_file(args.storage):
        nodes = load_nodes(args.storage)

    remove_old_nodes(nodes, datetime.timedelta(days=7))

    try:
        with open(args.maps, 'r') as maps:
            for line in maps.readlines():
                try:
                    AlfredParser.parse_line(line.strip(), nodes)
                except:
                    import traceback
                    # debug switch below
                    print(line)
                    traceback.print_exc()
                    continue
        nodes_json = render_nodelist(nodes)

    except IOError:
        exit('File ' + args.maps + ' not accessible')

    if args.storage:
        save_nodes(args.storage, nodes)

    global GLOBAL_NODES
    GLOBAL_NODES = nodes_json

    global PORT_NUMBER
    try:
        PORT_NUMBER = int(args.port)
    except ValueError:
        exit('Error: ' + args.port + ' is not a valid port-number')


if __name__ == '__main__':
    main()
    REGISTRY.register(CustomCollector())
    # Start up the server to expose the metrics.
    start_http_server(PORT_NUMBER)
    # Generate some requests.
    while True:
        time.sleep(10)
        main()
