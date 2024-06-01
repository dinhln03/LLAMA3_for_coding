#!/usr/bin/env python2

import rospy
from gnss_status_viewer import Status
from nmea_msgs.msg import Sentence
import sys
import copy


# Previous and current Status
prev = None
curr = None


def print_current_status(status):
    """
    Prints the current status
    :param status:
    :return:
    """
    print(status)
    # Move to the beginning of the previous line
    for i in range(str(status).count('\n') + 1):
        sys.stdout.write('\033[F')


def nmea_cb(msg):
    global prev
    global curr

    if prev is None:
        prev = Status(msg.sentence)
        return

    curr = Status(msg.sentence)
    if not curr.is_gga:
        return

    if prev != curr:
        status_change = Status.get_status_change(prev, curr)
        [ rospy.loginfo(s) for s in status_change ]
        n = max(map(lambda line: len(line), status_change))
        print(' ' * n)
    print_current_status(curr)

    prev = copy.deepcopy(curr)


rospy.init_node('gnss_status_viewer_node')

rospy.Subscriber('nmea_sentence', Sentence, nmea_cb)

rospy.spin()
