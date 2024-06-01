#!/usr/local/bin/python

# based on code by henryk ploetz
# https://hackaday.io/project/5301-reverse-engineering-a-low-cost-usb-co-monitor/log/17909-all-your-base-are-belong-to-us
# and the wooga office weather project
# https://blog.wooga.com/woogas-office-weather-wow-67e24a5338

import os, sys, fcntl, time, socket
from prometheus_client import start_http_server, Gauge, Summary, Counter
import requests


def callback_function(error, result):
    if error:
        print(error)
        return

    print(result)

def hd(d):
    return " ".join("%02X" % e for e in d)

def now():
    return int(time.time())

# Create a metric to track time spent and requests made.
decrypt_time = Summary('decrypt_time_seconds', 'Time spent decrypting')
# Decorate function with metric.
@decrypt_time.time()
def decrypt(key,  data):
    cstate = [0x48,  0x74,  0x65,  0x6D,  0x70,  0x39,  0x39,  0x65]
    shuffle = [2, 4, 0, 7, 1, 6, 5, 3]

    phase1 = [0] * 8
    for i, o in enumerate(shuffle):
        phase1[o] = data[i]

    phase2 = [0] * 8
    for i in range(8):
        phase2[i] = phase1[i] ^ key[i]

    phase3 = [0] * 8
    for i in range(8):
        phase3[i] = ( (phase2[i] >> 3) | (phase2[ (i-1+8)%8 ] << 5) ) & 0xff

    ctmp = [0] * 8
    for i in range(8):
        ctmp[i] = ( (cstate[i] >> 4) | (cstate[i]<<4) ) & 0xff

    out = [0] * 8
    for i in range(8):
        out[i] = (0x100 + phase3[i] - ctmp[i]) & 0xff

    return out



if __name__ == "__main__":
    """main"""

    # use lock on socket to indicate that script is already running
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        ## Create an abstract socket, by prefixing it with null.
        s.bind('\0postconnect_gateway_notify_lock')
    except socket.error, e:
        # if script is already running just exit silently
        sys.exit(0)

    key = [0xc4, 0xc6, 0xc0, 0x92, 0x40, 0x23, 0xdc, 0x96]
    fp = open(sys.argv[1], "a+b",  0)
    HIDIOCSFEATURE_9 = 0xC0094806
    set_report = "\x00" + "".join(chr(e) for e in key)
    fcntl.ioctl(fp, HIDIOCSFEATURE_9, set_report)

    values = {}
    stamp = now()
    notified = False

    # define Gauge metrice for temp and co2
    co2_metric = Gauge('co2_value', 'Current CO_2 Value from sensor')
    temperature_metric = Gauge('temperature_value', 'Current Temperature Value from sensor')

    # define loop counter
    loop_counter = Counter('loops_total', 'Number of loops to query the sensor for values')

    # Start up the server to expose the metrics.
    start_http_server(8000)

    while True:
        loop_counter.inc()
        data = list(ord(e) for e in fp.read(8))
        decrypted = decrypt(key, data)
        if decrypted[4] != 0x0d or (sum(decrypted[:3]) & 0xff) != decrypted[3]:
            print hd(data), " => ", hd(decrypted),  "Checksum error"
        else:
            op = decrypted[0]
            val = decrypted[1] << 8 | decrypted[2]
            values[op] = val

            if (0x50 in values) and (0x42 in values):
                co2 = values[0x50]
                tmp = (values[0x42]/16.0-273.15)

                # check if it's a sensible value
                # (i.e. within the measuring range plus some margin)
                if (co2 > 5000 or co2 < 0):
                    continue

                if now() - stamp > 10:
                    print "TMP %3.1f" % (tmp)
                    temperature_metric.set(tmp)
                    print "CO2 %4i" % (co2)
                    co2_metric.set(co2)
                    print ">>>"
                    stamp = now()
