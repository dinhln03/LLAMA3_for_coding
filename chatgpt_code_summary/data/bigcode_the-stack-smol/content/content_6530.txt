#!/usr/bin/env python
from multipledispatch import dispatch as Override
import rospy
import threading
from std_msgs.msg import Float64

from araig_msgs.msg import BoolStamped
from base_classes.base_calculator import BaseCalculator

"""Compare data from one topic with one param
    pub_list = {"out_bool": "BoolStamped"}
    sub_list = {"in_float": "Float64"}
    rosparam
    inherit Base, only modify compare function"""
class compParam(BaseCalculator):
    _pub_topic = "/out_bool"
    _sub_topic = "/in_float"
    def __init__(self,
        sub_dict = {_sub_topic: Float64}, 
        pub_dict = {_pub_topic: BoolStamped},
        rosparam = None,
        tolerance = 0,
        rate = None):

        if rosparam == None:
            rospy.logerr(rospy.get_name() + ": Please provide rosparam")
        else:
            self.compare_param = rosparam

        self.tolerance = tolerance        

        super(compParam, self).__init__(
            sub_dict = sub_dict,
            pub_dict = pub_dict,
            rate = rate)

    @Override()
    def calculate(self):
        with BaseCalculator.LOCK[self._sub_topic]:
            current_vel = BaseCalculator.MSG[self._sub_topic]

        flag_test_ready = True
        if current_vel == None:
            flag_test_ready = False

        if flag_test_ready == True:
            msg = self.PubDict[self._pub_topic]()
            msg.header.stamp = rospy.Time.now()

            if abs(self.compare_param - current_vel.data) <= self.tolerance:
                msg.data = True
                
            else:
                msg.data = False

            self.PubDiag[self._pub_topic].publish(msg)