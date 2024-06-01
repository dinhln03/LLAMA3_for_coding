#!/usr/bin/env python3

import importlib
import logging
import os
import traceback

import pybullet_data
import rclpy
from rclpy import executors
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Empty

from pybullet_ros.function_exec_manager import FuncExecManager


class pyBulletRosWrapper(Node):
    """ROS wrapper class for pybullet simulator"""
    def __init__(self):

        super().__init__('pybullet_ros', 
            automatically_declare_parameters_from_overrides=True)

        ex = MultiThreadedExecutor()
        self.executor = ex

        # import pybullet
        self.pb = importlib.import_module('pybullet')

        # get from param server the frequency at which to run the simulation
        self.loop_rate = self.get_parameter('loop_rate').value  
        self.get_logger().info('Loop rate: {}'.format(self.loop_rate))
        # query from param server if gui is needed
        is_gui_needed = self.get_parameter('pybullet_gui').value
        # get from param server if user wants to pause simulation at startup
        self.pause_simulation = self.get_parameter('pause_simulation').value  
        print('\033[34m')
        # print pybullet stuff in blue
        physicsClient = self.start_gui(gui=is_gui_needed) # we dont need to store the physics client for now...
        # setup service to restart simulation
        self.create_service(Empty, 'reset_simulation', self.handle_reset_simulation)
        # setup services for pausing/unpausing simulation
        self.create_service(Empty, 'pause_physics', self.handle_pause_physics)
        self.create_service(Empty, 'unpause_physics', self.handle_unpause_physics)
        # get pybullet path in your system and store it internally for future use, e.g. to set floor
        self.pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        # create object of environment class for later use
        env_plugin = self.get_parameter('environment').value # default : plugins/environment.py
        plugin_import_prefix = self.get_parameter('plugin_import_prefix').value
        self.environment = getattr(importlib.import_module(f'{plugin_import_prefix}.{env_plugin}'), 'Environment')(self)
        # load robot URDF model, set gravity, and ground plane
        self.robot = self.init_pybullet_robot()
        self.connected_to_physics_server = None
        if not self.robot:
            self.connected_to_physics_server = False
            return # Error while loading urdf file
        else:
            self.connected_to_physics_server = True
        # get all revolute joint names and pybullet index
        rev_joint_index_name_dic, prismatic_joint_index_name_dic, fixed_joint_index_name_dic, link_names_to_ids_dic = self.get_properties()
        # import plugins dynamically
        self.plugins = []
        plugins = self.get_parameter('plugins').value
        if not plugins:
            self.get_logger().warn('No plugins found, forgot to set param plugins?')
        # return to normal shell color
        print('\033[0m')
        # load plugins
        for plugin in plugins:
            module_, class_ = plugin.split(':')
            params_ = {'module': module_, 'class': class_}
            self.get_logger().info('loading plugin: {} class from {}'.format(class_, module_))
            # create object of the imported file class
            obj = getattr(importlib.import_module(module_), class_)(self.pb, self.robot,
                          rev_joints=rev_joint_index_name_dic,
                          prism_joints=prismatic_joint_index_name_dic,
                          fixed_joints=fixed_joint_index_name_dic,
                          link_ids=link_names_to_ids_dic,
                          **params_)
            # store objects in member variable for future use
            self.plugins.append(obj)
            self.executor.add_node(obj)

        self.get_logger().info('pybullet ROS wrapper initialized')
        self.timer = self.create_timer(1.0 / self.loop_rate, self.wrapper_callback)

        self.executor.add_node(self)

        try:
            self.executor.spin()
        #except Exception as e:
        #    self.get_logger().error(traceback.format_exc())
        finally:
            self.executor.shutdown()
            self.destroy_node()
            for node in self.plugins:
                node.destroy_node()

    def wrapper_callback(self):
        self.pb.stepSimulation()
        if not self.connected_to_physics_server:
            self.pb.disconnect()

    def get_properties(self):
        """
        construct 3 dictionaries:
        - joint index to joint name x2 (1 for revolute, 1 for fixed joints)
        - link name to link index dictionary
        """
        rev_joint_index_name_dic = {}
        fixed_joint_index_name_dic = {}
        prismatic_joint_index_name_dic = {}
        link_names_to_ids_dic = {}
        for joint_index in range(0, self.pb.getNumJoints(self.robot)):
            info = self.pb.getJointInfo(self.robot, joint_index)
            # build a dictionary of link names to ids
            link_names_to_ids_dic[info[12].decode('utf-8')] = joint_index
            # ensure we are dealing with a revolute joint
            if info[2] == self.pb.JOINT_REVOLUTE:
                # insert key, value in dictionary (joint index, joint name)
                rev_joint_index_name_dic[joint_index] = info[1].decode('utf-8') # info[1] refers to joint name
            elif info[2] == self.pb.JOINT_FIXED:
                # insert key, value in dictionary (joint index, joint name)
                fixed_joint_index_name_dic[joint_index] = info[1].decode('utf-8') # info[1] refers to joint name
            elif info[2] == self.pb.JOINT_PRISMATIC:
                prismatic_joint_index_name_dic[joint_index] = info[1].decode('utf-8') # info[1] refers to joint name
        return rev_joint_index_name_dic, prismatic_joint_index_name_dic, fixed_joint_index_name_dic, link_names_to_ids_dic

    def handle_reset_simulation(self, req):
        """Callback to handle the service offered by this node to reset the simulation"""
        self.get_logger().info('reseting simulation now')
        self.pb.resetSimulation()
        return Empty()

    def start_gui(self, gui=True):
        """start physics engine (client) with or without gui"""
        if(gui):
            # start simulation with gui
            self.get_logger().info('Running pybullet with gui')
            self.get_logger().info('-------------------------')
            gui_options = self.get_parameter('gui_options').value # e.g. to maximize screen: options="--width=2560 --height=1440"
            return self.pb.connect(self.pb.GUI, options=gui_options)
        else:
            # start simulation without gui (non-graphical version)
            self.get_logger().info('Running pybullet without gui')
            # hide console output from pybullet
            self.get_logger().info('-------------------------')
            return self.pb.connect(self.pb.DIRECT)

    def init_pybullet_robot(self):
        """load robot URDF model, set gravity, ground plane and environment"""
        # get from param server the path to the URDF robot model to load at startup
        urdf_path = self.get_parameter('robot_urdf_path').value
        if urdf_path == None:
            self.get_logger().warn('mandatory param robot_urdf_path not set, will exit now')
            rclpy.shutdown()
        # test urdf file existance
        if not os.path.isfile(urdf_path):
            self.get_logger().error('param robot_urdf_path is set, but file does not exist : ' + urdf_path)
            rclpy.shutdown()
            return None
        # ensure urdf is not xacro, but if it is then make urdf file version out of it
        if 'xacro' in urdf_path:
            # remove xacro from name
            urdf_path_without_xacro = urdf_path[0:urdf_path.find('.xacro')]+urdf_path[urdf_path.find('.xacro')+len('.xacro'):]
            os.system(f'xacro {urdf_path} -o {urdf_path_without_xacro}')
            urdf_path = urdf_path_without_xacro
        # get robot spawn pose from parameter server
        robot_pose_x = self.get_parameter('robot_pose_x').value
        robot_pose_y = self.get_parameter('robot_pose_y').value
        robot_pose_z = self.get_parameter('robot_pose_z').value
        robot_pose_yaw = self.get_parameter('robot_pose_yaw').value
        robot_spawn_orientation = self.pb.getQuaternionFromEuler([0.0, 0.0, robot_pose_yaw])
        fixed_base = self.get_parameter('fixed_base').value
        # load robot from URDF model
        # user decides if inertia is computed automatically by pybullet or custom
        if self.get_parameter('use_inertia_from_file').value:
            # combining several boolean flags using "or" according to pybullet documentation
            urdf_flags = self.pb.URDF_USE_INERTIA_FROM_FILE | self.pb.URDF_USE_SELF_COLLISION
        else:
            urdf_flags = self.pb.URDF_USE_SELF_COLLISION
        # load environment
        self.get_logger().info('loading environment')
        self.environment.load_environment()
        # set no realtime simulation, NOTE: no need to stepSimulation if setRealTimeSimulation is set to 1
        self.pb.setRealTimeSimulation(0) # NOTE: does not currently work with effort controller, thats why is left as 0
        self.get_logger().info('loading urdf model: ' + urdf_path)
        # NOTE: self collision enabled by default
        return self.pb.loadURDF(urdf_path, basePosition=[robot_pose_x, robot_pose_y, robot_pose_z],
                                           baseOrientation=robot_spawn_orientation,
                                           useFixedBase=fixed_base, flags=urdf_flags)

    def handle_reset_simulation(self, req):
        """Callback to handle the service offered by this node to reset the simulation"""
        self.get_logger().info('reseting simulation now')
        # pause simulation to prevent reading joint values with an empty world
        self.pause_simulation = True
        # remove all objects from the world and reset the world to initial conditions
        self.pb.resetSimulation()
        # load URDF model again, set gravity and floor
        self.init_pybullet_robot()
        # resume simulation control cycle now that a new robot is in place
        self.pause_simulation = False
        return []

    def handle_pause_physics(self, req):
        """pause simulation, raise flag to prevent pybullet to execute self.pb.stepSimulation()"""
        self.get_logger().info('pausing simulation')
        self.pause_simulation = False
        return []

    def handle_unpause_physics(self, req):
        """unpause simulation, lower flag to allow pybullet to execute self.pb.stepSimulation()"""
        self.get_logger().info('unpausing simulation')
        self.pause_simulation = True
        return []

    def pause_simulation_function(self):
        return self.pause_simulation

def main():
    try:
        rclpy.init()
        pyBulletRosWrapper()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
