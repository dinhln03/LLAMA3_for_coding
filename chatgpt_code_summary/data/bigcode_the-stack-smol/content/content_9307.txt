import numpy as np
np.random.seed(0)

ADOPT = 0
OVERRIDE = 1
WAIT = 2

class Environment(object):
    def __init__(self, mining_powers, gammas, T):
        # relative mining strengths.
        self.mining_powers = mining_powers
        self.gammas = gammas
        self.num_miners = len(mining_powers)
        
        # termination parameters
        self.T = T

        # chain variables
        self.chain = ''
        self.starting_points = np.zeros(self.num_miners, dtype=np.int64)
        self.hidden_lengths = np.zeros(self.num_miners, dtype=np.int64)
    
    def reset(self):
        self.chain = ''
        self.starting_points = np.zeros(self.num_miners, dtype=np.int64)
        self.hidden_lengths = np.zeros(self.num_miners, dtype=np.int64)

    def getNextBlockWinner(self):
        winner = np.random.choice(np.arange(len(self.mining_powers)), p=self.mining_powers)
        self.hidden_lengths[winner] += 1
        return winner
    
    def adopt(self, player_index):
        _a, h = self.getState(player_index)
        self.starting_points[player_index] = len(self.chain)
        self.hidden_lengths[player_index] = 0
        return self.getState(player_index), (0, h)
    
    def wait(self, player_index):
        a, h = self.getState(player_index)
        if (a == self.T) or (h == self.T):
            return self.adopt(player_index)
        return self.getState(player_index), (0, 0)
    
    def override(self, player_index):
        a, h = self.getState(player_index)
        if a <= h:
            self.starting_points[player_index] = len(self.chain)
            self.hidden_lengths[player_index] = 0
            return self.getState(player_index), (0, 10)
        
        # chop chain to proper length
        self.chain = self.chain[:self.starting_points[player_index]]
        new_blocks = str(player_index) * a
        self.chain += new_blocks
        self.starting_points[player_index] = len(self.chain)
        self.hidden_lengths[player_index] = 0
        return self.getState(player_index), (a, 0)

    def getState(self, player_index):
        return (self.hidden_lengths[player_index], len(self.chain)-self.starting_points[player_index])
    
    def takeActionPlayer(self, player_index, action):
        if action == ADOPT:
            return self.adopt(player_index)
        elif action == OVERRIDE:
            return self.override(player_index)
        elif action == WAIT:
            return self.wait(player_index)
        else:
            raise KeyError('{} is not an action'.format(action))

if __name__ == "__main__":
    powers = [0.55, 0.45]
    gammas = [0.5, 0.5]
    env = Environment(powers, gammas, T=9)
    chain = ''
    for _ in range(1000):
        chain += str(env.getNextBlockWinner())
    print('p0', chain.count('0'), chain.count('0')/len(chain))
    print('p1', chain.count('1'), chain.count('1')/len(chain))
    print('p2', chain.count('2'), chain.count('2')/len(chain))
    