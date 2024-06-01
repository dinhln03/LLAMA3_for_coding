#!/usr/bin/env python3
# license removed for brevity
#策略 機械手臂 四點來回跑
import rospy
import os
import numpy as np
from std_msgs.msg import String
from ROS_Socket.srv import *
from ROS_Socket.msg import *
import math
import enum
import .ArmCommand.Hiwin_RT605_Arm_Command as ArmTask
##----Arm state-----------
Arm_state_flag = 0
Strategy_flag = 0
##----Arm status enum
class Arm_status(enum.IntEnum):
    Idle = 0
    Isbusy = 1
    Error = 2
    shutdown = 6
##-----------server feedback arm state----------
def Arm_state(req):
    global CurrentMissionType,Strategy_flag,Arm_state_flag
    Arm_state_flag = int('%s'%req.Arm_state)
    if Arm_state_flag  == Arm_status.Isbusy: #表示手臂忙碌
        Strategy_flag = False
        return(1)
    if Arm_state_flag  == Arm_status.Idle: #表示手臂準備
        Strategy_flag = True
        return(0)
    if Arm_state_flag  == Arm_status.shutdown: #表示程式中斷
        Strategy_flag = 6
        return(6)
def arm_state_server():
    #rospy.init_node(NAME)
    s = rospy.Service('arm_state',arm_state, Arm_state) ##server arm state
    #rospy.spin() ## spin one
##-----------switch define------------##
class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

##------------class-------
class point():
    def __init__(self,x,y,z,pitch,roll,yaw):
        self.x = x
        self.y = y
        self.z = z
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw



##-------------------------strategy---------------------
##-----Mission 參數
GetInfoFlag = False
ExecuteFlag = False
GetKeyFlag = False
MotionSerialKey = []
MissionType_Flag = 0
MotionStep = 0
##-----手臂動作位置資訊
angle_SubCue = 0
LinePtpFlag = False
MoveFlag = False
PushBallHeight = 6
ObjAboveHeight = 10
SpeedValue = 10
MissionEndFlag = False
CurrentMissionType = 0
##---------------Enum---------------##
class ArmMotionCommand(enum.IntEnum):
    Arm_Stop = 0
    Arm_MoveToTargetUpside = 1
    Arm_MoveFowardDown = 2
    Arm_MoveVision = 3
    Arm_PushBall = 4
    Arm_LineUp = 5
    Arm_LineDown = 6
    Arm_Angle = 7
    Arm_StopPush = 8
class MissionType(enum.IntEnum):
    Get_Img = 0
    PushBall = 1
    Pushback = 2
    Mission_End = 3
##-----------switch define------------##
class pos():
    def __init__(self, x, y, z, pitch, roll, yaw):
        self.x = 0
        self.y = 36.8
        self.z = 11.35
        self.pitch = -90
        self.roll = 0
        self.yaw = 0
class Target_pos():
    def __init__(self, x, y, z, pitch, roll, yaw):
        self.x = 0
        self.y = 36.8
        self.z = 11.35
        self.pitch = -90
        self.roll = 0
        self.yaw = 0
class TargetPush_pos():
    def __init__(self, x, y, z, pitch, roll, yaw):
        self.x = 0
        self.y = 36.8
        self.z = 11.35
        self.pitch = -90
        self.roll = 0
        self.yaw = 0
class Item():
    def __init__(self,x,y,label):
        self.x = x
        self.y = y
        self.label = label

def Mission_Trigger():
    if GetInfoFlag == True and GetKeyFlag == False and ExecuteFlag == False:
        GetInfo_Mission()
    if GetInfoFlag == False and GetKeyFlag == True and ExecuteFlag == False:
        GetKey_Mission()
    if GetInfoFlag == False and GetKeyFlag == False and ExecuteFlag == True:
        Execute_Mission()
def GetInfo_Mission():
    global GetInfoFlag,GetKeyFlag,ExecuteFlag

    #Billiards_Calculation()

    GetInfoFlag = False
    GetKeyFlag = True
    ExecuteFlag = False
def GetKey_Mission():
    global GetInfoFlag,GetKeyFlag,ExecuteFlag,MotionKey,MotionSerialKey

    Mission = Get_MissionType()
    MissionItem(Mission)
    MotionSerialKey = MotionKey
    GetInfoFlag = False
    GetKeyFlag = False
    ExecuteFlag = True
def Get_MissionType():
    global MissionType_Flag,CurrentMissionType
    for case in switch(MissionType_Flag): #傳送指令給socket選擇手臂動作
        if case(0):
            Type = MissionType.PushBall
            MissionType_Flag +=1
            break
        if case(1):
            Type = MissionType.Pushback
            MissionType_Flag -=1
            break
    CurrentMissionType = Type
    return Type
def MissionItem(ItemNo):
    global MotionKey
    Key_PushBallCommand = [\
        ArmMotionCommand.Arm_MoveToTargetUpside,\
        ArmMotionCommand.Arm_LineDown,\
        ArmMotionCommand.Arm_PushBall,\
        ArmMotionCommand.Arm_LineUp,\
        ArmMotionCommand.Arm_Stop,\
        ]
    Key_PushBackCommand = [\
        ArmMotionCommand.Arm_MoveVision,\
        ArmMotionCommand.Arm_Stop,\
        ArmMotionCommand.Arm_StopPush,\
        ]
    for case in switch(ItemNo): #傳送指令給socket選擇手臂動作
        if case(MissionType.PushBall):
            MotionKey = Key_PushBallCommand
            break
        if case(MissionType.Pushback):
            MotionKey = Key_PushBackCommand
            break
    return MotionKey
def Execute_Mission():
    global GetInfoFlag,GetKeyFlag,ExecuteFlag,MotionKey,MotionStep,MotionSerialKey,MissionEndFlag,CurrentMissionType,Strategy_flag,Arm_state_flag
    if Arm_state_flag == Arm_status.Idle and Strategy_flag == True:
        Strategy_flag = False
        if MotionKey[MotionStep] == ArmMotionCommand.Arm_Stop:
            if MissionEndFlag == True:
                CurrentMissionType = MissionType.Mission_End
                GetInfoFlag = False
                GetKeyFlag = False
                ExecuteFlag = False
                print("Mission_End")
            elif CurrentMissionType == MissionType.PushBall:
                GetInfoFlag = False
                GetKeyFlag = True
                ExecuteFlag = False
                MotionStep = 0
                print("PushBall")
            else:
                GetInfoFlag = True
                GetKeyFlag = False
                ExecuteFlag = False
                MotionStep = 0
        else:
            MotionItem(MotionSerialKey[MotionStep])
            MotionStep += 1
def MotionItem(ItemNo):
    global angle_SubCue,SpeedValue,PushFlag,LinePtpFlag,MissionEndFlag
    SpeedValue = 5
    for case in switch(ItemNo): #傳送指令給socket選擇手臂動作
        if case(ArmMotionCommand.Arm_Stop):
            MoveFlag = False
            print("Arm_Stop")
            break
        if case(ArmMotionCommand.Arm_StopPush):
            MoveFlag = False
            PushFlag = True #重新掃描物件
            print("Arm_StopPush")
            break
        if case(ArmMotionCommand.Arm_MoveToTargetUpside):
            pos.x = 10
            pos.y = 36.8
            pos.z = 11.35
            pos.pitch = -90
            pos.roll = 0
            pos.yaw = 10
            MoveFlag = True
            LinePtpFlag = False
            SpeedValue = 10
            print("Arm_MoveToTargetUpside")
            break
        if case(ArmMotionCommand.Arm_LineUp):
            pos.z = ObjAboveHeight
            MoveFlag = True
            LinePtpFlag = True
            SpeedValue = 5
            print("Arm_LineUp")
            break
        if case(ArmMotionCommand.Arm_LineDown):
            pos.z = PushBallHeight
            MoveFlag = True
            LinePtpFlag = True
            SpeedValue = 5
            print("Arm_LineDown")
            break
        if case(ArmMotionCommand.Arm_PushBall):
            pos.x = -10
            pos.y = 36.8
            pos.z = 11.35
            pos.pitch = -90
            pos.roll = 0
            pos.yaw = -10
            SpeedValue = 10   ##待測試up
            MoveFlag = True
            LinePtpFlag = False
            print("Arm_PushBall")
            break
        if case(ArmMotionCommand.Arm_MoveVision):
            pos.x = 0
            pos.y = 36.8
            pos.z = 11.35
            pos.pitch = -90
            pos.roll = 0
            pos.yaw = 0
            SpeedValue = 10
            MoveFlag = True
            LinePtpFlag = False
            ##任務結束旗標
            MissionEndFlag = True
            print("Arm_MoveVision")
            break
        if case(ArmMotionCommand.Arm_MoveFowardDown):
            pos.x = 0
            pos.y = 36.8
            pos.z = 11.35
            pos.pitch = -90
            pos.roll = 0
            pos.yaw = 0
            MoveFlag = True
            LinePtpFlag = False
            print("Arm_MoveFowardDown")
            break
        if case(): # default, could also just omit condition or 'if True'
            print ("something else!")
            # No need to break here, it'll stop anyway
    if MoveFlag == True:
        if LinePtpFlag == False:
            print('x: ',pos.x,' y: ',pos.y,' z: ',pos.z,' pitch: ',pos.pitch,' roll: ',pos.roll,' yaw: ',pos.yaw)
            #strategy_client_Arm_Mode(0,1,0,30,2)#action,ra,grip,vel,both
            ArmTask.strategy_client_Arm_Mode(2,1,0,SpeedValue,2)#action,ra,grip,vel,both
            ArmTask.strategy_client_pos_move(pos.x,pos.y,pos.z,pos.pitch,pos.roll,pos.yaw)
        elif LinePtpFlag == True:
            #strategy_client_Arm_Mode(0,1,0,40,2)#action,ra,grip,vel,both
            print('x: ',pos.x,' y: ',pos.y,' z: ',pos.z,' pitch: ',pos.pitch,' roll: ',pos.roll,' yaw: ',pos.yaw)
            ArmTask.strategy_client_Arm_Mode(3,1,0,SpeedValue,2)#action,ra,grip,vel,both
            ArmTask.strategy_client_pos_move(pos.x,pos.y,pos.z,pos.pitch,pos.roll,pos.yaw)
    #action: ptp line
    #ra : abs rel
    #grip 夾爪
    #vel speed
    #both : Ctrl_Mode
##-------------strategy end ------------
def myhook():
    print ("shutdown time!")
if __name__ == '__main__':
    argv = rospy.myargv()
    rospy.init_node('strategy', anonymous=True)
    GetInfoFlag = True #Test no data
    arm_state_server()
    ArmTask.strategy_client_Arm_Mode(0,1,0,20,2)#action,ra,grip,vel,both
    while 1:
        Mission_Trigger()
        if CurrentMissionType == MissionType.Mission_End:
            ArmTask.rospy.on_shutdown(myhook)
            ArmTask.rospy.spin()
    rospy.spin()
