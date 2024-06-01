# the TestEnv environment is used to simply simulate the network
from flow.envs import TestEnv

# the Experiment class is used for running simulations
from flow.core.experiment import Experiment

# the base network class
from flow.networks import Network
from flow.envs.base import Env

# all other imports are standard
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.controllers import IDMController
from flow.core.params import InFlows
from flow.core.params import NetParams
from flow.core.params import TrafficLightParams
from flow.core.params import InitialConfig
from flow.core.params import EnvParams

from flow.controllers import IDMController, RLController, StaticLaneChanger

from gym.spaces.box import Box
import numpy as np
import collections

# create some default parameters parameters
HORIZON = 3000
env_params = EnvParams(
        horizon=HORIZON,
        sims_per_step=1,
        warmup_steps=0,
        additional_params={
            "max_accel": 3,
            "max_decel": -2,
            "target_velocity": 20,
            "lane_change_duration": 4,
            "num_rl": 5,
        })
initial_config = InitialConfig(edges_distribution=['highway_0'])

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    # lane_change_controller=(StaticLaneChanger, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=1621,
        model="SL2015",
        lc_impatience="0.1",
        lc_time_to_impatience="1.0"
    ))
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    lane_change_controller=(StaticLaneChanger, {}),
    # routing_controller=(HighwayRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=256,
        model="SL2015",
        lc_impatience="0.1",
        lc_time_to_impatience="1.0"
    ),
    num_vehicles=0)

from flow.core.params import SumoParams

sim_params = SumoParams(
        sim_step=0.2,
        render=True,
        lateral_resolution=1.0,
        restart_instance=True,
    )

import os

inflow = InFlows()
inflow.add(veh_type="human",
           edge="WC",
           # depart_lane="best",
           depart_lane=1,
           arrivalLane=0,
           probability=0.1,
           depart_speed="random",
           )
inflow.add(veh_type="human",
           edge="WC",
           # depart_lane="best",
           depart_lane=0,
           arrivalLane=1,
           probability=0.1,
           depart_speed="random",
           )
inflow.add(veh_type="human",
           edge="EC",
           # depart_lane="best",
           # vehs_per_hour=2000,
           depart_lane=1,
           arrivalLane=0,
           probability=0.1,
           depart_speed="random",
           )
inflow.add(veh_type="human",
           edge="EC",
           # depart_lane="best",
           # vehs_per_hour=2000,
           depart_lane=0,
           arrivalLane=1,
           probability=0.1,
           depart_speed="random",
           )
inflow.add(
    veh_type="rl",
    edge="WC",
    vehs_per_hour=100,
    depart_lane="free",
    depart_speed=5)

net_params = NetParams(
    template={
    "net":"/home/rong/Safe-RL-for-Driving/traci_pedestrian_crossing/pedcrossing.net.xml",
    # features associated with the routes vehicles take
    "vtype": "/home/rong/Safe-RL-for-Driving/traci_pedestrian_crossing/pedcrossing.add.xml",
    # 和下方specify_routes一致
    "rou":"/home/rong/Safe-RL-for-Driving/traci_pedestrian_crossing/data/pedcrossing.rou.xml",
    "trip":"/home/rong/Safe-RL-for-Driving/traci_pedestrian_crossing/pedestrians.trip.xml"
    },
    inflows=inflow,
)

# specify the edges vehicles can originate on
initial_config = InitialConfig(
    edges_distribution=["WC"]
)

tl_logic = TrafficLightParams(baseline=False)
phases = [{"duration": "100000", "state": "GGGGr"},
          {"duration": "4", "state": "yyyyr"},
          {"duration": "10", "state": "rrrrG"},
          {"duration": "10", "state": "rrrrr"}]
tl_logic.add("C", phases=phases, programID="custom", offset="0")


# specify the routes for vehicles in the network
class PedCrossing(Network):

    def specify_routes(self, net_params):
        return {'EC': ['EC', 'CW'],
                'WC': ['WC', 'CE']}


class MoveXYPedEnv(Env):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        # 环境相关
        self.activeRequest = False
        self.greenTimeSoFar = 0
        # minimum green time for the vehicles
        self.MIN_GREEN_TIME = 15
        # the first phase in tls plan. see 'pedcrossing.tll.xml'
        self.VEHICLE_GREEN_PHASE = 0
        self.PEDESTRIAN_GREEN_PHASE = 2
        # the id of the traffic light (there is only one). This is identical to the
        # id of the controlled intersection (by default)
        self.TLSID = 'C'
        # pedestrian edges at the controlled intersection
        self.WALKINGAREAS = [':C_w0', ':C_w1']
        self.CROSSINGS = [':C_c0']
    #     Move xy相关
        self.num_lanes = max(self.k.network.num_lanes(edge)
                             for edge in self.k.network.get_edge_list())
        self.visible = []
        self.stuck = False
        # variables used to sort vehicles by their initial position plus
        # distance traveled
        self.prev_pos = dict()
        self.absolute_position = dict()

        # maximum number of controlled vehicles
        self.num_rl = env_params.additional_params["num_rl"]

        # queue of rl vehicles waiting to be controlled
        self.rl_queue = collections.deque()

        # names of the rl vehicles controlled at any step
        self.rl_veh = []

        # used for visualization: the vehicles behind and after RL vehicles
        # (ie the observed vehicles) will have a different color
        self.leader = []
        self.follower = []

    @property
    def action_space(self):
        """See class definition."""
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lb = [1, -0.2] * self.num_rl
        ub = [2, 0.2] * self.num_rl
        # print("num_rl_vehicles:", self.num_rl)
        return Box(np.array(lb), np.array(ub), dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        # print("observation sapce shape: ", 4 * self.num_rl *
        #            self.num_lanes + self.num_rl)
        return Box(
            low=-1000,
            high=3000,
            shape=(4 * self.num_rl *
                   self.num_lanes + 2 * self.num_rl, ),
            dtype=np.float32)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        reward = 0

        # rl 车辆向前，并惩罚停止
        rl_velocity = np.array(self.k.vehicle.get_speed(self.rl_veh))
        target_vel = self.env_params.additional_params['target_velocity']
        max_cost = np.array([target_vel] * self.num_rl)
        max_cost = np.linalg.norm(max_cost)

        cost = rl_velocity - target_vel
        cost = np.linalg.norm(cost)
        # epsilon term (to deal with ZeroDivisionError exceptions)
        eps = np.finfo(np.float32).eps
        reward += max(max_cost - cost, 0) / (max_cost + eps)

        gain = 0.5
        thresh = 0.3
        penalize = len(rl_velocity[rl_velocity < thresh])
        reward -= gain * penalize

        # punish excessive lane changes by reducing the reward by a set value
        # every time an rl car changes lanes (10% of max reward)
        for veh_id in self.rl_veh:
            if self.k.vehicle.get_last_lc(veh_id) == self.time_counter:
                reward -= 10

        if self.stuck:
            reward -= 100
        # print("reward: ", reward)
        return reward

    def _apply_rl_actions(self, actions):
        """See class definition."""
        acceleration = actions[::2]
        direction = actions[1::2]

        # represents vehicles that are allowed to change lanes
        # non_lane_changing_veh = []
        # non_lane_changing_veh = \
        #     [self.time_counter <=
        #      self.env_params.additional_params["lane_change_duration"]
        #      + self.k.vehicle.get_last_lc(veh_id)
        #      for veh_id in self.rl_veh]
        # # vehicle that are not allowed to change have their directions set to 0
        # print(non_lane_changing_veh)
        # direction[non_lane_changing_veh] = \
        #     np.array([0] * sum(non_lane_changing_veh))
        for i, veh_id in enumerate(self.rl_veh):
            if self.time_counter <= self.env_params.additional_params["lane_change_duration"]\
                    + self.k.vehicle.get_last_lc(veh_id):
                direction[i] = 0
            x, y = self.k.vehicle.kernel_api.vehicle.getPosition(veh_id)
            print(x, y)
            print("edgeID", self.k.vehicle.get_edge(veh_id))
            print("lane", self.k.vehicle.get_lane(veh_id))
            self.k.vehicle.kernel_api.vehicle.moveToXY(vehID=veh_id,
                                                       edgeID="highway_1",
                                                       lane=1,
                                                       x=x+acceleration[i],
                                                       y=y+direction[i],
                                                       keepRoute=2)

        for x in np.nditer(direction, op_flags=['readwrite']):
            if x > 0.7:
                x[...] = 1
            elif x < -0.7:
                x[...] = -1
            else:
                x[...] = 0

        # print("actions:", actions)
        # print("veh id: ", self.rl_veh)
        # print("acceleration: ", acceleration)
        # print("direction", direction)

        # self.k.vehicle.apply_acceleration(self.rl_veh, acc=acceleration)
        # self.k.vehicle.apply_lane_change(self.rl_veh, direction=direction)

    def get_state(self):
        """See class definition."""
        obs = [
            0
            for _ in range(4 * self.num_rl * self.num_lanes + 2 * self.num_rl)
        ]
        # print("rl veh id: ", self.rl_veh)
        self.visible = []
        self.update_veh_id()
        speeds = []
        for i, rl_id in enumerate(self.rl_veh):
            # x, y = self.k.vehicle.kernel_api.vehicle.getPosition(rl_id)
            # print(x, y)
            # print("edgeID", self.k.vehicle.get_edge(rl_id))
            # print("lane", self.k.vehicle.get_lane(rl_id))
            # self.k.vehicle.kernel_api.vehicle.moveToXY(vehID=[rl_id, rl_id], edgeID="highway_1", lane=1, x=600, y=134)
            # add the speed for the ego rl vehicle
            x = self.k.vehicle.get_x_by_id(rl_id)
            if x == -1001:
                continue
            speed = self.k.vehicle.get_speed(rl_id)
            obs[-2*i - 1] = speed
            speeds.append(speed)
            obs[-2*i - 2] = x

            # if rl_id not in self.k.vehicle.get_ids():
            #     print("not in:", rl_id)
            #     self.additional_command()
            # normalizers
            max_length = self.k.network.length()
            max_speed = self.k.network.max_speed()

            # set to 1000 since the absence of a vehicle implies a large
            # headway
            headway = [1] * self.num_lanes
            tailway = [1] * self.num_lanes
            vel_in_front = [0] * self.num_lanes
            vel_behind = [0] * self.num_lanes

            lane_leaders = self.k.vehicle.get_lane_leaders(rl_id)
            lane_followers = self.k.vehicle.get_lane_followers(rl_id)
            lane_headways = self.k.vehicle.get_lane_headways(rl_id)
            lane_tailways = self.k.vehicle.get_lane_tailways(rl_id)
            headway[0:len(lane_headways)] = lane_headways
            tailway[0:len(lane_tailways)] = lane_tailways

            for j, lane_leader in enumerate(lane_leaders):
                if lane_leader != '':
                    lane_headways[j] /= max_length
                    vel_in_front[j] = self.k.vehicle.get_speed(lane_leader) \
                        / max_speed
                    self.visible.extend([lane_leader])
            for j, lane_follower in enumerate(lane_followers):
                if lane_follower != '':
                    lane_headways[j] /= max_length
                    vel_behind[j] = self.k.vehicle.get_speed(lane_follower) \
                        / max_speed
                    self.visible.extend([lane_follower])

            # add the headways, tailways, and speed for all lane leaders
            # and followers
            obs[4*self.num_lanes*i:4*self.num_lanes*(i+1)] = \
                np.concatenate((headway, tailway, vel_in_front, vel_behind))

        # if len(speeds) > 3:
        #     self.stuck = True
        #     for speed in speeds:
        #         if speed != 0:
        #             self.stuck = False
        obs = np.array(obs)
        # print("observation: ", obs)
        # print("observation shape: ", obs.shape)
        np.clip(obs, -1000, 3000, out=obs)
        return obs

    def additional_command(self):
        # 红绿灯相关
        # decide wether there is a waiting pedestrian and switch if the green
        # phase for the vehicles exceeds its minimum duration
        if not self.activeRequest:
            self.activeRequest = self.checkWaitingPersons()
        if self.k.kernel_api.trafficlight.getPhase(self.TLSID) == self.VEHICLE_GREEN_PHASE:
            self.greenTimeSoFar += 1
            if self.greenTimeSoFar > self.MIN_GREEN_TIME:
                # check whether someone has pushed the button

                if self.activeRequest:
                    # switch to the next phase
                    self.k.kernel_api.trafficlight.setPhase(
                        self.TLSID, self.VEHICLE_GREEN_PHASE + 1)
                    # reset state
                    self.activeRequest = False
        # MOVE XY相关
        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)

        # update the "absolute_position" variable
        for veh_id in self.k.vehicle.get_ids():
            this_pos = self.k.vehicle.get_x_by_id(veh_id)

            if this_pos == -1001:
                # in case the vehicle isn't in the network
                self.absolute_position[veh_id] = -1001
            else:
                change = this_pos - self.prev_pos.get(veh_id, this_pos)
                self.absolute_position[veh_id] = \
                    (self.absolute_position.get(veh_id, this_pos) + change) \
                    % self.k.network.length()
                self.prev_pos[veh_id] = this_pos
        return

    def update_veh_id(self):
        # add rl vehicles that just entered the network into the rl queue
        for veh_id in self.k.vehicle.get_rl_ids():
            if veh_id not in list(self.rl_queue) + self.rl_veh:
                self.rl_queue.append(veh_id)

        # remove rl vehicles that exited the network
        for veh_id in list(self.rl_queue):
            if veh_id not in self.k.vehicle.get_rl_ids() or veh_id not in self.k.vehicle.get_ids():
                self.rl_queue.remove(veh_id)
        for veh_id in self.rl_veh:
            if veh_id not in self.k.vehicle.get_rl_ids() or veh_id not in self.k.vehicle.get_ids():
                # print("rm veh_id", veh_id)
                self.rl_veh.remove(veh_id)

        # fil up rl_veh until they are enough controlled vehicles
        while len(self.rl_queue) > 0 and len(self.rl_veh) < self.num_rl:
            rl_id = self.rl_queue.popleft()
            self.rl_veh.append(rl_id)
            # print("add rl_veh:", rl_id)
        # print("update_veh_id, self.rl_veh:", self.rl_veh)

    def checkWaitingPersons(self):
        """check whether a person has requested to cross the street"""

        # check both sides of the crossing
        for edge in self.WALKINGAREAS:
            peds = self.k.kernel_api.edge.getLastStepPersonIDs(edge)
            # check who is waiting at the crossing
            # we assume that pedestrians push the button upon
            # standing still for 1s
            for ped in peds:
                if (self.k.kernel_api.person.getWaitingTime(ped) == 1 and
                        self.k.kernel_api.person.getNextEdge(ped) in self.CROSSINGS):
                    numWaiting = self.k.kernel_api.trafficlight.getServedPersonCount(self.TLSID, self.PEDESTRIAN_GREEN_PHASE)
                    print("%s: pedestrian %s pushes the button (waiting: %s)" %
                          (self.k.kernel_api.simulation.getTime(), ped, numWaiting))
                    return True
        return False

    def step(self, rl_actions):
        """Advance the environment by one step.

        Assigns actions to autonomous and human-driven agents (i.e. vehicles,
        traffic lights, etc...). Actions that are not assigned are left to the
        control of the simulator. The actions are then used to advance the
        simulator by the number of time steps requested per environment step.

        Results from the simulations are processed through various classes,
        such as the Vehicle and TrafficLight kernels, to produce standardized
        methods for identifying specific network state features. Finally,
        results from the simulator are used to generate appropriate
        observations.

        Parameters
        ----------
        rl_actions : array_like
            an list of actions provided by the rl algorithm

        Returns
        -------
        observation : array_like
            agent's observation of the current environment
        reward : float
            amount of reward associated with the previous state/action pair
        done : bool
            indicates whether the episode has ended
        info : dict
            contains other diagnostic information from the previous action
        """
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # perform acceleration actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    action = self.k.vehicle.get_acc_controller(
                        veh_id).get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel)

            # perform lane change actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
                direction = []
                for veh_id in self.k.vehicle.get_controlled_lc_ids():
                    target_lane = self.k.vehicle.get_lane_changing_controller(
                        veh_id).get_action(self)
                    direction.append(target_lane)
                self.k.vehicle.apply_lane_change(
                    self.k.vehicle.get_controlled_lc_ids(),
                    direction=direction)

            # perform (optionally) routing actions for all vehicles in the
            # network, including RL and SUMO-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) \
                        is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(
                        veh_id)
                    routing_actions.append(route_contr.choose_route(self))

            self.k.vehicle.choose_routes(routing_ids, routing_actions)

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)

            # update the colors of vehicles
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()

            # stop collecting new simulation steps if there is a collision
            if crash:
                break

            # render a frame
            self.render()

        states = self.get_state()

        # collect information of the state of the network based on the
        # environment class used
        self.state = np.asarray(states).T

        # collect observation new state associated with action
        next_observation = np.copy(states)

        # test if the environment should terminate due to a collision or the
        # time horizon being met
        done = (self.time_counter >= self.env_params.warmup_steps +
                self.env_params.horizon) or self.stuck

        if done:
            print("done")
            if self.stuck:
                print("stuck")
            else:
                print("time up")
        # compute the info for each agent
        infos = {}

        # compute the reward
        if self.env_params.clip_actions:
            rl_clipped = self.clip_actions(rl_actions)
            reward = self.compute_reward(rl_clipped, fail=crash)
        else:
            reward = self.compute_reward(rl_actions, fail=crash)

        return next_observation, reward, done, infos

    def reset(self):
        """See parent class.

        This also includes updating the initial absolute position and previous
        position.
        """
        self.rl_queue.clear()
        self.rl_veh.clear()
        obs = super().reset()
        print("reset")
        for veh_id in self.k.vehicle.get_ids():
            self.absolute_position[veh_id] = self.k.vehicle.get_x_by_id(veh_id)
            self.prev_pos[veh_id] = self.k.vehicle.get_x_by_id(veh_id)

        self.leader = []
        self.follower = []
        return obs


if __name__ == "__main__":
    flow_params = dict(
        exp_tag='template',
        env_name=MoveXYPedEnv,
        network=PedCrossing,
        simulator='traci',
        sim=sim_params,
        env=env_params,
        net=net_params,
        veh=vehicles,
        initial=initial_config,
        tls=tl_logic,
    )

    # number of time steps
    flow_params['env'].horizon = 10000
    exp = Experiment(flow_params)

    # run the sumo simulation
    _ = exp.run(1)
