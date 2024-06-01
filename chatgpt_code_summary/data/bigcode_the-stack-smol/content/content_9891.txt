#!/usr/bin/python3

# apt install libnetfilter-queue-dev

import os
import random
import string
import time
from multiprocessing import Pool
from netfilterqueue import NetfilterQueue
from scapy.all import *

SINGLE_QUEUE = False

if SINGLE_QUEUE:
    nfqueue_number = 1
else:
    nfqueue_number = 4

def setup():
    k_module = "modprobe br_netfilter"
    os.system(k_module)

    if SINGLE_QUEUE:
        iptables_rule = "iptables -A FORWARD -j NFQUEUE --queue-num %d -m physdev --physdev-in ens38" % (nfqueue_number - 1)
    else:
        iptables_rule = "iptables -A FORWARD -j NFQUEUE --queue-balance 0:%d -m physdev --physdev-in ens38" % (nfqueue_number - 1)
    

    print("Adding iptable rules : ")
    print(iptables_rule)
    os.system(iptables_rule)

    print("Setting ipv4 forward settings : ")
    os.system("sysctl net.ipv4.ip_forward=1")

def change_payload(packet, load):
    packet[Raw].load = load

    del packet[IP].len
    del packet[IP].chksum
    del packet[TCP].chksum
    #python2
    #return packet.__class__(packet)

    #python3
    return packet.__bytes__()

def slack_chars(payload, source, target, finalize=False):
    if source in payload["data"]:
        payload["diff"] += len(source) - len(target)
        payload["data"] = payload["data"].replace(source, target)

    if finalize:
        slacks = [b"\r\nAccept-Encoding: gzip, deflate", b"\r\nConnection: Keep-Alive"]
        payload["diff"] += len(slacks[0])
        payload["data"] = payload["data"].replace(slacks[0], b"")

        for slack in slacks[1:]:
            if payload["diff"] < 0:
                payload["diff"] += len(slack)
                payload["data"] = payload["data"].replace(slack, b"")

        if payload["diff"] > 7:
            header = b"\r\nID: "
            stuff = b"".join(bytes(random.choice(string.ascii_uppercase + string.digits), "ascii") for _ in range(payload["diff"] - len(header)))
            payload["data"] = payload["data"][:-4:] + header + stuff
        else:
            payload["data"] = payload["data"][:-4:] + b" ".join(b"" for _ in range(payload["diff"]))

        payload["data"] = payload["data"] + b"\r\n\r\n"

        payload["diff"] = 0

    return payload

def callback(payload):
    print(payload)
    try:
        data = payload.get_payload()
        pkt = IP(data)
    
        if isinstance(pkt.payload, TCP):
            if isinstance(pkt[TCP].payload, Raw):
                raw_payload = pkt[TCP].load
                
                if raw_payload.startswith(b"GET ") or raw_payload.startswith(b"POST "):
                    if b"Windows NT 6.1" in raw_payload:
                        wrap_payload = {"diff": 0, "data": raw_payload}
                        if b"; WOW64; Trident/" not in raw_payload:
                            wrap_payload = slack_chars(wrap_payload, b"; Trident/", b"; WOW64; Trident/")
                        wrap_payload = slack_chars(wrap_payload, b"Accept-Language: ja-JP\r\n", b"Accept-Language: ko-KR\r\n")
                        wrap_payload = slack_chars(wrap_payload, b"Accept-Language: en-US\r\n", b"Accept-Language: ko-KR\r\n", finalize=True)

                        raw_payload = wrap_payload["data"]
                        new_pkt = change_payload(pkt, raw_payload)
                        payload.set_payload(new_pkt)
    except Exception as e:
        print(e)
    finally:
        payload.accept()

def main():
    setup()
    if SINGLE_QUEUE:
        start(0)
    else:
        p = Pool(nfqueue_number)
        try:
            p.map_async(start, [x for x in range(nfqueue_number)]).get(999999999)
            p.close()
        except KeyboardInterrupt:
            p.terminate()
    print("Flushing iptables.")
    os.system('iptables -F')
    os.system('iptables -X')


def start(queue_num):
    nfqueue = NetfilterQueue()
    nfqueue.bind(queue_num, callback)

    try:
        nfqueue.run(block=True)
    finally:
        nfqueue.unbind()

if __name__ == "__main__":
    main()
