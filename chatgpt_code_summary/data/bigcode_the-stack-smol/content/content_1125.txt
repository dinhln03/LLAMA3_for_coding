#! /usr/bin/env python
import rospy
from std_msgs.msg import Int32

def cb(message):
    rospy.loginfo(message.data*2)
    print (rospy.loginfo)
    if message.data%2 == 0:

      print ('0')

    elif message.data%2 != 0 :
      
      print('1')

if __name__ == '__main__':
    rospy.init_node('twice')
    sub = rospy.Subscriber('count_up', Int32, cb)
    rospy.spin()
