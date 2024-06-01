#!/usr/bin/env python

import sys

last_pkt_num = -1

daystart_pkt_num = -1
daystart_recv_time = -1
daystart_hwrecv_time = -1

dayend_pkt_num = -1
dayend_recv_time = -1
dayend_hwrecv_time = -1

def process_line(line):
    global last_pkt_num
    global daystart_pkt_num, daystart_recv_time, daystart_hwrecv_time
    global dayend_pkt_num, dayend_recv_time, dayend_hwrecv_time

    parts = line.split()
    pkt_num = long(parts[1])
    sent_time = long(parts[3])
    recv_time = long(parts[5])
    hw_recv_time = long(parts[7])

    # read in the first line
    if (daystart_pkt_num == -1):
        last_pkt_num = pkt_num

        daystart_pkt_num = pkt_num
        daystart_recv_time = recv_time
        daystart_hwrecv_time = hw_recv_time

        dayend_pkt_num = pkt_num
        dayend_recv_time = recv_time
        dayend_hwrecv_time = hw_recv_time

        return

    # skip through the day, looking for a gap
    if (pkt_num == last_pkt_num + 1):
        last_pkt_num = pkt_num

        dayend_pkt_num = pkt_num
        dayend_recv_time = recv_time
        dayend_hwrecv_time = hw_recv_time

        return

    # we found a gap
    dstr = "D {} pkts long, {} us (utime), {} us (hw)".format(
        dayend_pkt_num - daystart_pkt_num,
        dayend_recv_time - daystart_recv_time,
        dayend_hwrecv_time - daystart_hwrecv_time)
    print(dstr)

    nstr = "\t\t\t\t\t\t\t\tN {} pkts long, {} us (utime), {} us (hw)".format(
        pkt_num - dayend_pkt_num,
        recv_time - dayend_recv_time,
        hw_recv_time - dayend_hwrecv_time)
    print(nstr)

    last_pkt_num = pkt_num

    daystart_pkt_num = pkt_num
    daystart_recv_time = recv_time
    daystart_hwrecv_time = hw_recv_time
    

def main(argv):
    if (len(argv) == 1):
        fin = sys.stdin
    else:
        fin = open(argv[1])

    while 1:
        try:
            line = fin.readline()
        except KeyboardInterrupt:
            break

        if not line:
            break

        process_line(line)

if __name__ == "__main__":
    main(sys.argv)
