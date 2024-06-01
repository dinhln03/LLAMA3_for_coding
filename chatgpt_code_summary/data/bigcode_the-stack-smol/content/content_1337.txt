#!/usr/bin/env python
# file trying to apply and test the pid controller on carla.

import glob
import os
import sys
import time
import matplotlib.pyplot as plt
from PID_controller import PID
import numpy as np
import speed_profile_reader as spr

try:
    sys.path.append(glob.glob('../**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time


class TestData:

    def __init__(self, total_duration, time_increment):
        self._iter_num = 0
        self.time = np.empty([int(total_duration / time_increment) + 1, 1])
        self.setpoint = np.empty([int(total_duration / time_increment) + 1, 1])
        self.actual_velocity = np.empty([int(total_duration / time_increment) + 1, 1])
        self.error = np.empty([int(total_duration / time_increment) + 1, 1])

    def append_data(self, t, sp, vel, error):
        self.time[self._iter_num] = t
        self.setpoint[self._iter_num] = sp
        self.actual_velocity[self._iter_num] = vel
        self.error[self._iter_num] = error
        self._iter_num+=1

    def plot(self):
        plt.figure()
        plt.plot(self.time, self.setpoint)
        plt.plot(self.time, self.actual_velocity)
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title("PID Result")
        plt.figure()
        plt.plot(self.time, self.error, 'r--', label='error', alpha=0.75, linewidth=0.5)
        plt.plot(self.time, np.zeros(len(self.time)), 'k--', linewidth=0.5)
        plt.title("Controller Error")
        plt.show()

class DataInit:
    K = {
        "Kp": 0.055734,
        "Ki": 0.0114169,
        "Kd": .00006

        # For 10 m/s
        # "Kp": 0.055734,
        # "Ki": 0.0130169,
        # "Kd": .000006

        # "Kp": 1,
        # "Ki": 0.0112,
        # "Kd": 0.000006
    }
    total_duration = 20
    sampling_period = 0.025

def main():

    actor_list = []
    verboseIsEnabled = None
    try:
        """
        Section for starting the client and connecting to the server
        """
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        for arg in sys.argv:
            if (arg == '--verbose'):
                verboseIsEnabled = True

        if verboseIsEnabled:
            print('client version: %s' % client.get_client_version())
            print('server version: %s' % client.get_server_version())
            print('client to server connection status: {}'.format(client.get_server_version()))

            print('Retrieving the world data from server...')

        world = client.get_world()
        if verboseIsEnabled:
            print('{} \n'.format(world))

        """
        Section for retrieving the blueprints and spawn the actors
        """
        blueprint_library = world.get_blueprint_library()
        if verboseIsEnabled:
            print('\nRetrieving CARLA blueprint library...')
            print('\nobject: %s\n\nblueprint methods: %s\n\nblueprint list:' % (type(blueprint_library), dir(blueprint_library)) )
            for blueprint in blueprint_library:
                print(blueprint)

        audi_blueprint = blueprint_library.find('vehicle.audi.tt')
        print('\n%s\n' % audi_blueprint)

        color = '191,191,191'
        audi_blueprint.set_attribute('color', color)

        transform = carla.Transform(
			carla.Location(
                x=10.5, y=-1.8,
            z=38.5),carla.Rotation(yaw=0.0)
		)

        vehicleEgo = world.spawn_actor(audi_blueprint, transform)
        actor_list.append(vehicleEgo)
        print('created %s' % vehicleEgo.type_id)

        color = random.choice(audi_blueprint.get_attribute('color').recommended_values)
        audi_blueprint.set_attribute('color', color)

        """
        Section for initializing the PID testing
        """
        user_input_sp = None
        while (not isinstance(user_input_sp, int)) and (not isinstance(user_input_sp, float)):
            user_input_sp = input('Enter the desired Setpoint:\n')
        data = TestData(DataInit.total_duration, DataInit.sampling_period)
        start = time.time()

        print('\nStarting test:\n\n' + 'Time(s) current_vel(m/s) setpoint_vel(m/s) throttle(%) pid_demand')
        time.sleep(2.5)
        print('.................................................................\n')
        time.sleep(1)

        # raise SystemExit

        p = PID(
                DataInit.K['Kp'], 
                DataInit.K['Ki'],
                DataInit.K['Kd']
                )
        p.setPoint(user_input_sp)
        p.Integrator_min = -5
        p.Integrator_max = 40
        pid = 0
        for _ in range(int(DataInit.total_duration / DataInit.sampling_period) + 1):
            measurement_value = vehicleEgo.get_velocity().x
            vehicleEgo.apply_control(carla.VehicleControl(pid)) if 1 > pid > 0 else vehicleEgo.apply_control(carla.VehicleControl(1))
            if 0 > pid: vehicleEgo.apply_control(carla.VehicleControl(brake=abs(pid)))
            pid = p.update(measurement_value)
            data.append_data(round(time.time() - start, 2), p.getSetPoint(), round(vehicleEgo.get_velocity().x, 5), p.getError())
            time.sleep(DataInit.sampling_period)

            print('%0.3f\t%0.2f\t\t\t%0.2f\t\t%0.2f\t%0.2f' % (time.time() - start,
                                                                vehicleEgo.get_velocity().x,
                                                                p.set_point,
                                                                vehicleEgo.get_control().throttle,
                                                                pid))

        data.plot()
        print('\nError Mean (Steady State):\n' + 
            str(round(np.absolute(np.mean(data.error[data.error.shape[0]/2:data.error.shape[0]])), 5)*100) + 
            '%\n')

    finally:
        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')

if __name__ == '__main__':

    main()
