#!/usr/bin/env python

# this node will be implemented on the master node

# this is a test script for drive motor 
# in function of stop and front lights detection
# this script will be implemented in another node 

# import libraries
import rospy,sys,time,atexit,numpy
from std_msgs.msg import String,Int16MultiArray

# define variables
avoidingVehicle = False
array = Int16MultiArray()
array.data = []
controlPub = rospy.Publisher("cmd",Int16MultiArray,queue_size=1)

def turnOffMotors():
	array.data = [0,0,0,0]
	controlPub.publish(array)

def setSpeed(motor1,motor2):
	if motor1 == 0 and motor2 == 0:
		turnOffMotors()
	else:
		array.data = [motor1,motor2,0,0]
		controlPub.publish(array)
	
def avoidVehicle(): 
	global avoidingVehicle
	turnOffMotors()
	avoidingVehicle = False 

def callback(data):
	global avoidingVehicle
	rospy.loginfo(rospy.get_caller_id() +" Led control String received: %s",data.data)
	if data.data == "stop" :
		turnOffMotors()
	elif (data.data == "front" and avoidingVehicle == False):
		avoidingVehicle = True
		avoidVehicle()
	elif data.data == "w":
		setSpeed(150,150)
	elif data.data == "s":
		turnOffMotors()
	
def led_control():
	rospy.init_node('led_control',anonymous=True)
	rospy.Subscriber('led_control_topic',String,callback)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':
	led_control()
