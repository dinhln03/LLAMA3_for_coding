#!/usr/bin/env python
from __future__ import print_function, absolute_import
import re
import subprocess
import os
import time
import argparse
import sys


class SmartDevice(object):

    smartcmdfmt = ['sudo', 'smartctl', '-f', 'brief', '-A', '/dev/{dev}']

    def __init__(self, dev):
        self.dev = dev
        self.attrcmd = [x.format(dev=dev) for x in self.smartcmdfmt]

    def attributes(self):
        try:
            out = subprocess.check_output(self.attrcmd)
        except (OSError, subprocess.CalledProcessError) as err:
            print('Error running command: {0}'.format(err), file=sys.stderr)
            return
        for line in out.split("\n"):
            res = re.match('\s*(?P<id>\d+)\s+(?P<name>[\w-]+)\s+'
                           '(?P<flags>[POSRCK-]{6})\s+'
                           '(?P<value>\d+)\s+(?P<worst>\d+)\s+'
                           '(?P<thres>\d+)\s+(?P<fail>[\w-]+)\s+'
                           '(?P<raw_value>\d+)', line)
            if not res:
                continue
            yield res.groupdict()


def dev_exists(dev):
    return os.path.exists('/dev/{0}'.format(dev))

def get_filelist(dirname, pattern):
    return [f for f in os.listdir(dirname) if re.match(pattern, f)]

def expand_devices(devlist):
    expanded_devlist = []
    for dev in devlist:
        if dev == 'autodetect':
            expanded_devlist.extend(get_filelist('/dev', r'^sd[a-z]+$'))
        else:
            expanded_devlist.append(dev)
    return sorted(list(set(expanded_devlist)))

def smartmon_loop(devices, hostname, interval):
    while True:
        for dev in devices:
            if dev_exists(dev):
                for attr in SmartDevice(dev).attributes():
                    print('PUTVAL "{hostname}/smart-{dev}'
                          '/absolute-{attr_id:03d}_{attr_name}"'
                          ' interval={interval:d} N:{value:d}'
                          .format(hostname=hostname, dev=dev,
                                  attr_id=int(attr['id']),
                                  attr_name=attr.get('name'),
                                  interval=int(interval),
                                  value=int(attr['raw_value'])))
        time.sleep(interval)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dev', nargs='*',
                        help='devices to check (default: autodetect)')
    parser.add_argument('-H', '--hostname', type=str,
                        help='override hostname provided by collectd',
                        default=os.environ.get('COLLECTD_HOSTNAME'))
    parser.add_argument('-i', '--interval', type=int,
                        help='override interval provided by collectd',
                        default=int(float(os.environ.get('COLLECTD_INTERVAL', 300))))
    parser.add_argument('-c', '--dont-check-devs',
                        action='store_true', default=False,
                        help='do not check devices existence at startup')
    args = parser.parse_args()

    hostname = (args.hostname
                or subprocess.check_output(['hostname', '-f']).strip())
    if len(hostname) == 0:
        parser.error('unable to detect hostname')
    interval = max(args.interval, 5)
    if len(args.dev) == 0:
        devices = expand_devices(['autodetect'])
    else:
        devices = expand_devices(args.dev)

    if not args.dont_check_devs:
        for dev in devices:
            if not dev_exists(dev):
                parser.error('device "/dev/{0}" does not exist'.format(dev))

    try:
        smartmon_loop(devices, hostname, interval)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
