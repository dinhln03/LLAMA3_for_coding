#!/usr/bin/env python
'''command long'''
import threading
import time, os
import math
from pymavlink import mavutil

from MAVProxy.modules.lib import mp_module

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt 
from threading import Thread

import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


class CmdlongModule(mp_module.MPModule):
    def __init__(self, mpstate):
        super(CmdlongModule, self).__init__(mpstate, "cmdlong")
        self.add_command('setspeed', self.cmd_do_change_speed, "do_change_speed")
        self.add_command('setyaw', self.cmd_condition_yaw, "condition_yaw")
        self.add_command('offboard', self.offboard_mode, "offboard")
        self.add_command('p_mode', self.position_mode, "p_mode")
        self.add_command('m_mode', self.manual_mode, "m_mode")
        self.add_command('a_mode', self.altitude_mode, "a_mode")
        self.add_command('takeoff2', self.cmd_takeoff_2, "takeoff2")
        self.add_command('takeoff3', self.takeoff_3, "takeoff3")
        self.add_command('music',self.music,"music")
        self.add_command('land2', self.land_2, "land2")
        self.add_command('fly', self.fly, "fly")
        self.add_command('x', self.x, "x")
        self.add_command('y', self.y, "y")
        self.add_command('z', self.z, "z")
        self.add_command('h', self.h, "h")
        self.add_command('yaw', self.yaw, "yaw")
        self.add_command('takeoff', self.cmd_takeoff, "takeoff")
        self.add_command('velocity', self.cmd_velocity, "velocity")
        self.add_command('position', self.cmd_position, "position")
        self.add_command('st', self.start_position_thread, "start_position_thread")
        self.add_command('attitude', self.cmd_attitude, "attitude")
        self.add_command('cammsg', self.cmd_cammsg, "cammsg")
        self.add_command('camctrlmsg', self.cmd_camctrlmsg, "camctrlmsg")
        self.add_command('posvel', self.cmd_posvel, "posvel")
        self.add_command('parachute', self.cmd_parachute, "parachute",
                         ['<enable|disable|release>'])
        self.add_command('long', self.cmd_long, "execute mavlink long command",
                         self.cmd_long_commands())
        self.dis_max = 0
        self.dis_min = 100
        self.dis_diff = self.dis_max - self.dis_min
        
        
        self.svo_x_max = 0
        self.svo_x_min = 0
        self.svo_y_max = 0
        self.svo_y_min = 0
        self.x_diff = self.svo_x_max - self.svo_x_min
        self.y_diff = self.svo_y_max - self.svo_y_min
        
        
        self.list_x = []
        self.list_y = []
        self.list_z = []
        self.svo_x = 0
        self.svo_y = 0
        self.svo_z = 0
        
        #thread_obj = Thread(target = self.show_svo_2d)
        #thread_obj = Thread(target = self.show_svo)
        #thread_obj.setDaemon(True)
        #thread_obj.start()
        
        
    def cmd_long_commands(self):
        atts = dir(mavutil.mavlink)
        atts = filter( lambda x : x.lower().startswith("mav_cmd"), atts)
        ret = []
        for att in atts:
            ret.append(att)
            ret.append(str(att[8:]))
        return ret

    def cmd_takeoff(self, args):
        '''take off'''
        if ( len(args) != 1):
            print("Usage: takeoff ALTITUDE_IN_METERS")
            return
        
        if (len(args) == 1):
            altitude = float(args[0])
            print("Take Off started")
            self.master.mav.command_long_send(
                self.settings.target_system,  # target_system
                mavutil.mavlink.MAV_COMP_ID_SYSTEM_CONTROL, # target_component
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, # command
                0, # confirmation
                0, # param1
                0, # param2
                0, # param3
                0, # param4
                0, # param5
                0, # param6
                altitude) # param7

    def cmd_parachute(self, args):
        '''parachute control'''
        usage = "Usage: parachute <enable|disable|release>"
        if len(args) != 1:
            print(usage)
            return

        cmds = {
            'enable'  : mavutil.mavlink.PARACHUTE_ENABLE,
            'disable' : mavutil.mavlink.PARACHUTE_DISABLE,
            'release' : mavutil.mavlink.PARACHUTE_RELEASE
            }
        if not args[0] in cmds:
            print(usage)
            return
        cmd = cmds[args[0]]
        self.master.mav.command_long_send(
            self.settings.target_system,  # target_system
            0, # target_component
            mavutil.mavlink.MAV_CMD_DO_PARACHUTE,
            0,
            cmd,
            0, 0, 0, 0, 0, 0)

    def cmd_camctrlmsg(self, args):
        '''camctrlmsg'''
        
        print("Sent DIGICAM_CONFIGURE CMD_LONG")
        self.master.mav.command_long_send(
            self.settings.target_system,  # target_system
            0, # target_component
            mavutil.mavlink.MAV_CMD_DO_DIGICAM_CONFIGURE, # command
            0, # confirmation
            10, # param1
            20, # param2
            30, # param3
            40, # param4
            50, # param5
            60, # param6
            70) # param7

    def cmd_cammsg(self, args):
        '''cammsg'''
  
        print("Sent DIGICAM_CONTROL CMD_LONG")
        self.master.mav.command_long_send(
            self.settings.target_system,  # target_system
            0, # target_component
            mavutil.mavlink.MAV_CMD_DO_DIGICAM_CONTROL, # command
            0, # confirmation
            10, # param1
            20, # param2
            30, # param3
            40, # param4
            50, # param5
            60, # param6
            70) # param7

    def cmd_do_change_speed(self, args):
        '''speed value'''
        if ( len(args) != 1):
            print("Usage: speed SPEED_VALUE")
            return
        
        if (len(args) == 1):
            speed = float(args[0])
            print("SPEED %s" % (str(speed)))
            self.master.mav.command_long_send(
                self.settings.target_system,  # target_system
                mavutil.mavlink.MAV_COMP_ID_SYSTEM_CONTROL, # target_component
                mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, # command
                0, # confirmation
                0, # param1
                speed, # param2 (Speed value)
                0, # param3
                0, # param4
                0, # param5
                0, # param6
                0) # param7

    def cmd_condition_yaw(self, args):
        '''yaw angle angular_speed angle_mode'''
        if ( len(args) != 3):
            print("Usage: yaw ANGLE ANGULAR_SPEED MODE:[0 absolute / 1 relative]")
            return
        
        if (len(args) == 3):
            angle = float(args[0])
            angular_speed = float(args[1])
            angle_mode = float(args[2])
            print("ANGLE %s" % (str(angle)))
            self.master.mav.command_long_send(
                self.settings.target_system,  # target_system
                mavutil.mavlink.MAV_COMP_ID_SYSTEM_CONTROL, # target_component
                mavutil.mavlink.MAV_CMD_CONDITION_YAW, # command
                0, # confirmation
                angle, # param1 (angle value)
                angular_speed, # param2 (angular speed value)
                0, # param3
                angle_mode, # param4 (mode: 0->absolute / 1->relative)
                0, # param5
                0, # param6
                0) # param7

    def cmd_velocity(self, args):
        '''velocity x-ms y-ms z-ms'''
        if (len(args) != 3):
            print("Usage: velocity x y z (m/s)")
            return

        if (len(args) == 3):
            x_mps = float(args[0])
            y_mps = float(args[1])
            z_mps = float(args[2])
            print("x:%f, y:%f, z:%f" % (x_mps, y_mps, z_mps))
            self.master.mav.set_position_target_local_ned_send(
                                      0,  # system time in milliseconds
                                      1,  # target system
                                      0,  # target component
                                      8,  # coordinate frame MAV_FRAME_BODY_NED
                                      455,      # type mask (vel only)
                                      0, 0, 0,  # position x,y,z
                                      x_mps, y_mps, z_mps,  # velocity x,y,z
                                      0, 0, 0,  # accel x,y,z
                                      0, 0)     # yaw, yaw rate
    
    
    def mavlink_packet(self, msg):
        type = msg.get_type()
            
        if type == 'DISTANCE_SENSOR':
            #print "distance find\n"
            #print isinstance(msg,subclass)
            #print msg.current_distance
            #print msg.__class__
            #self.console.set_status('distance','distance %s' % msg.current_distance)
            #print msg.current_distance
            if self.dis_max < msg.current_distance:
                self.dis_max = msg.current_distance
            if self.dis_min > msg.current_distance:
                self.dis_min = msg.current_distance
            self.dis_diff = self.dis_max - self.dis_min
            #self.msg.current_distance = 
        if type == 'SVO_POSITION_RAW':
            #self.svo_x = msg.position_x
            #self.svo_y = msg.position_y
            #self.svo_z = msg.position_z
            if self.svo_x_max < msg.position_x:
                self.svo_x_max = msg.position_x
            if self.svo_x_min > msg.position_x:
                self.svo_x_min = msg.position_x
            if self.svo_y_max < msg.position_y:
                self.svo_y_max = msg.position_y
            if self.svo_y_min > msg.position_y:
                self.svo_y_min = msg.position_y   
            self.x_diff = self.svo_x_max - self.svo_x_min
            self.y_diff = self.svo_y_max - self.svo_y_min
            #print self.dis_max
            #print self.dis_min
        elif type == 'LOCAL_POSITION_NED':
            self.console.set_status('position_ned_x','position_x %s' % msg.x)
            self.svo_x = msg.x
            #print type(self.svo_x)
            #self.console.set_status('position_ned_y','position_y %s' % msg.y)
            self.svo_y = msg.y
            #print (svo_y)
            #self.console.set_status('position_ned_z','position_ned %s' % msg.z)
            self.svo_z = msg.z
    def show_svo_2d(self):
        fig = plt.figure()
        #self.ax = p3.Axes3D(fig)
        self.ax = fig.add_subplot(1, 1, 1)
        num = 0
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('2D Test')

        self.ax.set_xlim([-1, 1])

        self.ax.set_ylim([-1, 1])

        self.num = 0
        #self.lineData = self.ax.scatter(1, 1, c = 'b', marker = '.')
        self.lineData, = self.ax.plot([],[])
        line_ani = animation.FuncAnimation(fig, self.update_lines_2d,self.Gen_RandLine_2d, 
                                           interval=100, blit=False)
        
        plt.show()



    def show_svo(self):
        fig = plt.figure()
        #self.ax = p3.Axes3D(fig)
        self.ax = fig.add_subplot(1, 1, 1, projection="3d")
        num = 0
        self.ax.set_xlabel('X')
        num = 0
        self.ax.set_xlabel('X')
        self.ax.set_xlim3d([-1.0, 1.0])
        self.ax.set_ylabel('Y')
        self.ax.set_ylim3d([-1.0, 1.0])
        self.ax.set_zlabel('Z')
        self.ax.set_zlim3d([-1.0, 1.0])
        self.ax.set_title('3D Test')
        self.num = 0
        #line_ani = animation.FuncAnimation(fig, self.update_lines,self.Gen_RandLine, 
        #                                   interval=10, blit=False)
        self.lineData = self.ax.scatter([1], [1], [1], c = 'b', marker = '.')
        line_ani = animation.FuncAnimation(fig, self.update_lines,self.Gen_RandLine, 
                                           interval=10, blit=False)
        
        plt.show()
        
    def data_stream(self):
        pass
    
    
    
    def Gen_RandLine_2d(self):
        
        if len(self.list_x)<200:
            self.list_x.append(self.svo_x)  
            self.list_y.append(self.svo_y)
            self.list_z.append(self.svo_z)
        else:
            self.list_x.append(self.svo_x)  
            self.list_x = self.list_x[1:]
            self.list_y.append(self.svo_y)
            self.list_y = self.list_y[1:]
            self.list_z.append(self.svo_z)
            self.list_z = self.list_z[1:]
            
        #for i in range(2):
        #list_x = self.svo_x
        #list_y = self.svo_y
        
        self.list_x.append(float(self.svo_x))
        self.list_y.append(float(self.svo_y))
        #self.list_z.append(float(self.svo_z))
        lineData = [self.list_x,self.list_y]
        #lineData = [list_x,list_y]
        #print type(list_x)
        #print lineData
        #time.sleep(0.02)
        #self.ax.set_zlim(min(data[2]), max(data[2]))
        #lineData = [self.list_x,self.list_y,self.list_z]
        yield lineData
        
    def update_lines_2d(self,data):
        #print "data",data
    
        #lineData = self.ax.scatter(data[0], data[1], data[2], c = 'b', marker = '.')
        #self.lineData.set_data([(data[0], data[1])])
        self.lineData.set_xdata(data[0])
        self.lineData.set_ydata(data[1])
        self.num = self.num + 1
        
        #self.ax.set_xlim(min(data[0]), max(data[0]))
        #self.ax.set_ylim(min(data[1]), max(data[1]))
        
        if self.num == 100:
            #self.ax.cla()
            #print self.num
            self.num = 0
            self.ax.set_xlim(min(data[0])-1, max(data[0])+1)
            self.ax.set_ylim(min(data[1])-1, max(data[1])+1)
            
        return self.lineData,
        
        
    def Gen_RandLine(self):
        '''
        if len(self.list_x)<70:
            self.list_x.append(self.svo_x)  
            self.list_y.append(self.svo_y)
            self.list_z.append(self.svo_z)
        else:
            self.list_x.append(self.svo_x)  
            self.list_x = self.list_x[1:]
            self.list_y.append(self.svo_y)
            self.list_y = self.list_y[1:]
            self.list_z.append(self.svo_z)
            self.list_z = self.list_z[1:]
         '''   
        #for i in range(2):
        list_x = self.svo_x
        list_y = self.svo_y
        list_z = self.svo_z
        
        #self.list_x.append(float(self.svo_x))
        #self.list_y.append(float(self.svo_y))
        #self.list_z.append(float(self.svo_z))
        #lineData = [self.list_x,self.list_y,self.list_z]
        lineData = [[list_x],[list_y],[list_z]]
        #print type(list_x)
        #print lineData
        #self.ax.set_xlim(min(data[0]), max(data[0]))
        #self.ax.set_ylim(min(data[1]), max(data[1]))
        #self.ax.set_zlim(min(data[2]), max(data[2]))
        #lineData = [self.list_x,self.list_y,self.list_z]
        yield lineData
    def update_lines(self,data):
        #print "data",data
    
        #lineData = self.ax.scatter(data[0], data[1], data[2], c = 'b', marker = '.')
        self.lineData.set_offsets([(data[0], data[1])])
        #self.lineData.set_data([data[0], data[1]])
        self.lineData.set_3d_properties([data[2]], "z")
        self.num = self.num + 1
        if self.num == 200:
            #self.ax.cla()
            #print self.num
            self.num = 0
            self.ax.set_xlabel('X')
            #self.ax.set_xlim3d([-1.0, 1.0])
            self.ax.set_ylabel('Y')
            #self.ax.set_ylim3d([-1.0, 1.0])
            self.ax.set_zlabel('Z')
            #self.ax.set_zlim3d([-1.0, 1.0])
            self.ax.set_title('3D Test')
            print "xdiff",self.x_diff
            print "ydiff",self.y_diff
        #lineData = ax.scatter(data[0], data[1], data[2], c = 'b', marker = '.')
        #plt.pause(0.01)
        #ax = p3.Axes3D(fig)
        return self.lineData



    def position_mode(self,args):
        print "position mode!!!!!!!!!!!!!!!!!"
        self.list_x = []
        self.list_y = []
        self.list_z = []
        #self.start_position_thread(1)
        time.sleep(0.5)
        #self.master.set_mode(221,6,0)
        self.master.set_mode(129,3,0)
        self.dis_max = 0
        self.dis_min = 100
        
        self.svo_x_max = 0
        self.svo_x_min = 0
        self.svo_y_max = 0
        self.svo_y_min = 0
        self.x_diff = self.svo_x_max - self.svo_x_min
        self.y_diff = self.svo_y_max - self.svo_y_min
    def manual_mode(self,args):
        print "manual mode!!!!!!!!!!!!!!!!!"
        print self.master.__class__
        #self.start_position_thread(1)
        #time.sleep(0.5)
        #self.master.set_mode(221,6,0)
        self.master.set_mode(129,1,0)
        self.v_z = float(args[0])
    def altitude_mode(self,args):
        print "altitude mode!!!!!!!!!!!!!!!!!"
        #self.start_position_thread(1)
        #time.sleep(0.5)
        #self.master.set_mode(221,6,0)
        self.master.set_mode(129,2,0)
        #self.v_z = float(370)
        #self.dis_max = 0
        #self.dis_min = 100
        
    def offboard_mode(self,args):
        print "offboard!!!!!!!!!!!!!!!!!"
        #self.cmd_position_2(1)
        self.start_offboard_thread(1)
        time.sleep(0.5)
        self.master.set_mode(221,6,0)
        #self.master.set_mode(1,3,0)
        
    def cmd_takeoff_2(self, args):
        '''position z-m'''
        if (len(args) != 1):
            print("Usage: position z (meters)")
            return

        if (len(args) == 1):
        #    x_m = float(0)
        #    y_m = float(0)
            z_m = float(args[0])
        #    print("x:%f, y:%f, z:%f" % (x_m, y_m, z_m))
            self.master.mav.set_position_target_local_ned_send(
                                      0,  # system time in milliseconds
                                      1,  # target system
                                      0,  # target component
                                      8,  # coordinate frame MAV_FRAME_BODY_NED
                                      5571,     # type mask (pos only)
                                      0, 0, z_m,  # position x,y,z
                                      0, 0, 0,  # velocity x,y,z
                                      0, 0, 0,  # accel x,y,z
                                      0, 0)     # yaw, yaw rate
    
    
    def takeoff_3(self,args):
        self.type_mask = 5571
        #self.type_mask = 3576
        
        self.x_m = float(0)
        self.y_m = float(0)
        self.z_m = float(1.5)
        
        self.v_x = float(0)
        self.v_y = float(0)
        self.v_z = float(0)
        #self.cmd_position([1,1,1])
    
    def music(self,args):
        self.master.mav.command_long_send(
                self.settings.target_system,  # target_system
                1, # target_component
                0, # command
                1, # confirmation
                0, # param1
                0, # param2 (Speed value)
                0, # param3
                0, # param4
                0, # param5
                0, # param6
                0) # param7
        print self.settings.target_system
        print mavutil.mavlink.MAV_COMP_ID_SYSTEM_CONTROL
    
    def land_2(self,args):
        self.type_mask = 9671
        
        self.v_x = float(0)
        self.v_y = float(0)
        self.v_z = float(0)
    
    #def h(self,args):
    #    self.type_mask = 1479
    #    self.v_x = float(0)
    #    self.v_y = float(0)
    #    self.v_z = float(0)
    def x(self,args):
        #print self.master.flightmode
        self.type_mask = 1479
        if self.master.flightmode == "POSCTL":
            #print self.master
            self.v_x = float(args[0])*0.5
        elif self.master.flightmode == "ALTCTL":
            #print self.master
            self.v_x = float(args[0])*1
        elif self.master.flightmode == "MANUAL":
            #print self.master
            self.v_x = float(args[0])*1
        #self.v_z = -4
        self.button = 1
        
    def y(self,args):
        self.type_mask = 1479
        if self.master.flightmode == "POSCTL":
            self.v_y = float(args[0])*0.5
        elif self.master.flightmode == "ALTCTL":
            self.v_y = float(args[0])*1
        elif self.master.flightmode == "MANUAL":
            self.v_y = float(args[0])*1
        
        #self.v_z = -4
        self.button = 1
    def z(self,args):
        self.type_mask = 1479
        #self.v_z = float(args[0])
        if self.master.flightmode == "POSCTL":
            self.v_z = self.v_z + int(args[0])
        elif self.master.flightmode == "ALTCTL":
            self.v_z = self.v_z + int(args[0])
        elif self.master.flightmode == "MANUAL":
            self.v_z = self.v_z + int(args[0])*0.1
        self.button = 1
    def yaw(self,args):
        self.type_mask = 1479
        #self.yaw_rate = float(float(args[0])*(math.pi/6.0))
        self.yaw_rate = float(args[0])*1.5 
        self.button = 1
        #time.sleep(0.5)
        #self.yaw_rate = float(0)
    def h(self,args):
        self.type_mask = 1479
        self.v_x = float(0)
        self.v_y = float(0)
        if self.master.flightmode == "POSCTL":
            self.v_z = float(args[0])
        elif self.master.flightmode == "ALTCTL":
            self.v_z = float(args[0])
        elif self.master.flightmode == "MANUAL":
            pass
        self.yaw_rate = float(0)
        self.button = 0
    def fly(self,args):
        self.type_mask = 1479
        self.v_x = float(1)  
        time.sleep(2)
        self.v_x = float(0)  
        self.v_y = float(1)  
        time.sleep(2)
        self.v_y = float(0)  
        self.v_x = float(-1)  
        time.sleep(2)
        self.v_x = float(0) 
        self.v_y = float(-1) 
        time.sleep(2)
        self.v_y = float(0) 
    
    def start_position_thread(self,args):
        thread_obj = threading.Thread(target=self._cmd_position_2)
        thread_obj.setDaemon(True)
        thread_obj.start()
        #pass
    def start_offboard_thread(self,args):
        thread_obj = threading.Thread(target=self._cmd_position_2_offboard)
        thread_obj.start()
    def _cmd_position_2_offboard(self):
        '''position x-m y-m z-m'''
        #if (len(args) != 3):
        #    print("Usage: position x y z (meters)")
        #    return

        #if (len(args) == 3):
        self.type_mask = 17863
        
        self.x_m = float(0)
        self.y_m = float(0)
        self.z_m = float(0)
        
        self.v_x = float(0)
        self.v_y = float(0)
        self.v_z = float(0)
        
        self.yaw_rate = float(0)
            #print("x:%f, y:%f, z:%f" % (x_m, y_m, z_m))
        while 1:
            time.sleep(0.05)
            #print "type_mask:%s\n" % self.type_mask
            #print "v_x:%s\n" % self.v_x
            #print "v_y:%s\n" % self.v_y
            #print "v_z:%s\n" % self.v_z
            #print "z_m:%s\n" % self.z_m
            #print "send idle"
            
            self.master.mav.set_position_target_local_ned_send(
                                      0,  # system time in milliseconds
                                      1,  # target system
                                      0,  # target component
                                      8,  # coordinate frame MAV_FRAME_BODY_NED
                                      self.type_mask,     # type mask (pos only) 42707
                                      self.x_m, self.y_m, self.z_m,  # position x,y,z
                                      self.v_x, self.v_y, self.v_z,  # velocity x,y,z
                                      0, 0, 0,  # accel x,y,z
                                      0, self.yaw_rate)     # yaw, yaw rate
            
            
            
    def _cmd_position_2(self):
        print "position2"
        '''position x-m y-m z-m'''
        #if (len(args) != 3):
        #    print("Usage: position x y z (meters)")
        #    return

        #if (len(args) == 3):
        #self.type_mask = 17863
        
        #self.x_m = float(0)
        #self.y_m = float(0)
        #self.z_m = float(0)
        
        self.v_x = 0
        self.v_y = 0
        self.v_z = 0
        
        self.yaw_rate = 0
        self.button = 0
            #print("x:%f, y:%f, z:%f" % (x_m, y_m, z_m))
        i = 0
        while 1:
            time.sleep(0.05)
            #print "type_mask:%s\n" % self.type_mask
            #print "v_x:%s\n" % self.v_x
            #print "v_y:%s\n" % self.v_y
            #print "v_z:%s\n" % self.v_z
            #print "z_m:%s\n" % self.z_m
            #print "send idle"
            self.master.mav.manual_control_send(self.master.target_system,
                                   self.v_x, self.v_y,
                                   self.v_z, self.yaw_rate,
                                   self.button)
            
            i = i + 1
            if 0:
            #if i == 100:
                print "x",(int(self.v_x))
                print "y",(int(self.v_y))
                print "z",(int(self.v_z))
                print "yaw",(int(self.yaw_rate))
                print "dis_diff",(self.dis_diff)
                print "x_diff",(self.x_diff)
                print "y_diff",(self.y_diff)
                print "button",self.button
                print "target",(self.master.target_system)
                i = 0
    def cmd_position3(self, args):
        '''position x-m y-m z-m'''
        if (len(args) != 3):
            print("Usage: position x y z (meters)")
            return

        if (len(args) == 3):
            x_m = float(args[0])
            y_m = float(args[1])
            z_m = float(args[2])
            print("x:%f, y:%f, z:%f" % (x_m, y_m, z_m))
            self.master.mav.set_position_target_local_ned_send(
                                      0,  # system time in milliseconds
                                      1,  # target system
                                      0,  # target component
                                      8,  # coordinate frame MAV_FRAME_BODY_NED
                                      1479,     # type mask (pos only)
                                      0, 0, 0,# position x,y,z
                                      x_m, y_m, z_m,  # velocity x,y,z
                                      0, 0, 0,  # accel x,y,z
                                      0, 0)     # yaw, yaw rate

    def cmd_position(self, args):
        '''position x-m y-m z-m'''
        if (len(args) != 3):
            print("Usage: position x y z (meters)")
            return

        if (len(args) == 3):
            x_m = float(args[0])
            y_m = float(args[1])
            z_m = float(args[2])
            print("x:%f, y:%f, z:%f" % (x_m, y_m, z_m))
            self.master.mav.set_position_target_local_ned_send(
                                      0,  # system time in milliseconds
                                      1,  # target system
                                      0,  # target component
                                      8,  # coordinate frame MAV_FRAME_BODY_NED
                                      3576,     # type mask (pos only)
                                      x_m, y_m, z_m,  # position x,y,z
                                      0, 0, 0,  # velocity x,y,z
                                      0, 0, 0,  # accel x,y,z
                                      0, 0)     # yaw, yaw rate

    def cmd_attitude(self, args):
        '''attitude q0 q1 q2 q3 thrust'''
        if len(args) != 5:
            print("Usage: attitude q0 q1 q2 q3 thrust (0~1)")
            return

        if len(args) == 5:
            q0 = float(args[0])
            q1 = float(args[1])
            q2 = float(args[2])
            q3 = float(args[3])
            thrust = float(args[4])
            att_target = [q0, q1, q2, q3]
            print("q0:%.3f, q1:%.3f, q2:%.3f q3:%.3f thrust:%.2f" % (q0, q1, q2, q3, thrust))
            self.master.mav.set_attitude_target_send(
                                      0,  # system time in milliseconds
                                      1,  # target system
                                      0,  # target component
                                      63, # type mask (ignore all except attitude + thrust)
                                      att_target, # quaternion attitude
                                      0,  # body roll rate
                                      0,  # body pich rate
                                      0,  # body yaw rate
                                      thrust)  # thrust

    def cmd_posvel(self, args):
        '''posvel mapclick vN vE vD'''
        ignoremask = 511
        latlon = None
        try:
            latlon = self.module('map').click_position
        except Exception:
            pass
        if latlon is None:
            print "set latlon to zeros"
            latlon = [0, 0]
        else:
            ignoremask = ignoremask & 504
            print "found latlon", ignoremask            
        vN = 0
        vE = 0
        vD = 0
        if (len(args) == 3):
            vN = float(args[0])
            vE = float(args[1])
            vD = float(args[2])
            ignoremask = ignoremask & 455

        print "ignoremask",ignoremask
        print latlon
        self.master.mav.set_position_target_global_int_send(
            0,  # system time in ms
            1,  # target system
            0,  # target component
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            ignoremask, # ignore
            int(latlon[0] * 1e7),
            int(latlon[1] * 1e7),
            10,
            vN, vE, vD, # velocity
            0, 0, 0, # accel x,y,z
            0, 0) # yaw, yaw rate

    def cmd_long(self, args):
        '''execute supplied command long'''
        if len(args) < 1:
            print("Usage: long <command> [arg1] [arg2]...")
            return
        command = None
        if args[0].isdigit():
            command = int(args[0])
        else:
            try:
                command = eval("mavutil.mavlink." + args[0])
            except AttributeError as e:
                try:
                    command = eval("mavutil.mavlink.MAV_CMD_" + args[0])
                except AttributeError as e:
                    pass

        if command is None:
            print("Unknown command long ({0})".format(args[0]))
            return

        floating_args = [ float(x) for x in args[1:] ]
        while len(floating_args) < 7:
            floating_args.append(float(0))
        self.master.mav.command_long_send(self.settings.target_system,
                                          self.settings.target_component,
                                          command,
                                          0,
                                          *floating_args)

def init(mpstate):
    '''initialise module'''
    return CmdlongModule(mpstate)
