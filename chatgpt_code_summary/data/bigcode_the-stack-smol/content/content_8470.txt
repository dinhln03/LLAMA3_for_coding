import os
import sys
import numpy as np
import torch
import pickle 

import logging
log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


class Graph4D():

    def __init__(self, num_envs=4096, env_size=(4,4), steps=128, save=False, data_root='./data/', num_categories=None, verbose=False):
        self.num_envs = num_envs
        self.steps = steps
        self.env_size = env_size
        self.save = False
        self.data_root = data_root
        self.num_categories = num_categories
        self.generate_data(verbose=verbose)
        log.info('''Generated Data: 
                    \t\t\t {} Environments...
                    \t\t\t {} env size...
                    \t\t\t {} steps in each... 
                    \t\t\t {} observable one hot categories... '''.format(
                        num_envs, env_size, steps, self.num_categories
                    ))


    def square_env(self):
        """
        Generate map where each vertex has a one hot categorical distribution
        Returns:
            (N,N,num_categories) matrix with one-hot categorical observations
        """
        env_size = self.env_size
        env = np.zeros((env_size[0], env_size[1], self.num_categories))
        for i in range(env_size[0]):
            # Randomly assign categories to each vertex in a row
            category = np.random.randint(0, self.num_categories, env_size[1])
            # One hot encode them
            env[i, np.arange(category.size), category] = 1
        
        return env

        
    def update_location_4way(self, env, loc):
        """
        Samples a valid four-way action and updates location 
        """
        length = env.shape[0]
        valid = False
        # print(loc, end=' --> ')
        while not valid:
            # Sample action
            action = np.random.randint(0, 4)
            # Move up
            if action == 0:
                if loc[0] - 1 >= 0:
                    # print('Moving up', end=' --> ')
                    loc[0] -= 1
                    valid = True
            # Move right
            elif action == 1:
                if loc[1] + 1 < length:
                    # print('Moving Right', end=' --> ')
                    loc[1] += 1
                    valid = True
            # Move down
            elif action == 2:
                if loc[0] + 1 < length:
                    # print('Moving Down', end=' --> ')
                    loc[0] += 1
                    valid = True
            # Move left
            elif action == 3:
                if loc[1] - 1 >= 0:
                    # print('Moving Left', end=' --> ')
                    loc[1] -= 1
                    valid = True
        
        # One hot encode action
        act = np.zeros(4)
        act[action] = 1
        return act, loc


    def trajectory_4way(self, env):
        """
        Generate trajectory of agent diffusing through 4-way connected graph
        At each point we sample the one-hot observation and take an action
        0 = up
        1 = right
        2 = down
        3 = left 

        Params:
            steps (int): Number of steps to take
            env (3d np array): environment in which to wander (NxNx(num_categories))
        Returns 
            Observations (steps, num_categories), Actions (steps, 4) 
        """    
        observations = np.zeros((self.steps, self.num_categories))
        actions = np.zeros((self.steps, 4))
        positions = np.zeros((self.steps, 2))

        loc = np.random.randint(0, env.shape[0], 2) # Initial Location

        for step in range(self.steps):
            positions[step] = loc
            obs = env[loc[0], loc[1]]                       # Observe scene
            action, loc = self.update_location_4way(env, loc)    # Sample action and new location
            observations[step] = obs    
            actions[step] = action

        return observations, actions, positions       
    
    def generate_data(self, verbose=False):
        """
        Generates N square environments and trajectories ((observation, action) pairs)
        for each environment

        Params:
            envs (int): number of environments to generate
            steps (int): how many steps an agent initially takes in each environment
            env_size (tuple): size of environment (should be something like (4,4), (9,9), etc...)
            save (bool): whether or not to save the dataset
        
        Returns:
            Dict of "environments, observations, actions", each corresponding to: 
                        environments: Array shape: (num_envs, env_size_x, env_size_y, num_categories), 
                        observations: Array shape: (num_envs, steps, num_categories),
                        actions: Array shape: (num_envs, steps, 4)
        """
        env_size = self.env_size
        if self.num_categories == None:
            self.num_categories = env_size[0] * env_size[1]
            
        self.environments = np.zeros((self.num_envs, env_size[0], env_size[1], self.num_categories))
        self.observations = np.zeros((self.num_envs, self.steps, self.num_categories)) 
        self.actions = np.zeros((self.num_envs, self.steps, 4))  
        self.positions = np.zeros((self.num_envs, self.steps, 2))
        
        for i in range(self.num_envs):
            env = self.square_env()                     # Generate new environment
            obs, acts, pos = self.trajectory_4way(env)     # Generate random walk for that environment 

            self.environments[i] = env
            self.observations[i] = obs
            self.actions[i] = acts
            self.positions[i] = pos

        self.data = {'environments': self.environments, 'observations': self.observations, 'actions': self.actions, 'positions': self.positions}
        if self.save:
            name = os.path.join(self.data_root, 'four_way_graph.pickle')
            with open(name, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
if __name__=='__main__':
    print('Generating 20 (8,8) environments with 256 random steps in each.')
    graph = Graph4D(num_envs=20, env_size=(8,8), steps=256)
    data = graph.data
    envs = graph.environments
    observations = graph.observations
    actions = graph.actions
    positions = graph.positions
    print('Envs,', envs.shape)
    print('Obs', observations.shape)
    print('Acts', actions.shape)
    print('Pos', positions.shape)