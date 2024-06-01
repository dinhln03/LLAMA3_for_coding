#!/bin/env python3

import os
import time
import socket
import pickle
from optparse import OptionParser


if __name__=='__main__':

    parser = OptionParser(usage="%prog: [options]")
    parser.add_option("", "--fname", dest="fname", type="string",
                        help="Send a file, can also put filename at end")
    parser.add_option("", "--mcast", dest="ip", type="string", default="224.1.1.5",
                        help="IP Address to send to [default=%default]")
    parser.add_option("", "--port", dest="port", type="int", default=10000,
                        help="IP Address to send to [default=%default]")

    (options, args) = parser.parse_args()

    fname = options.fname

    if fname == None:
        if len(args) > 0:
            fname = args[0]
        else:
            parser.print_help()
            exit(1)

    MCAST_GRP = options.ip
    MCAST_PORT = options.port
    MCAST_TTL = 2

    print("Sending file", fname,"mcast group", MCAST_GRP, "port", MCAST_PORT)

    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    send_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MCAST_TTL)

    segments = []
    ff = open(fname, 'rb')
    num = ff.read(1024)
    pnum = 0

    while len(num) > 0:
        segments.append({"size":len(num), "num":pnum,"data":num,"fname":fname})
        pnum += 1
        num = ff.read(1024)

    for packet in segments:
        packet["total"] = len(segments)

    while True:
        for packet in segments:
            send_sock.sendto(pickle.dumps(packet), (MCAST_GRP, MCAST_PORT))
            time.sleep(0.0001)
        print("Sent file")
        time.sleep(0.100)
