import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';
SEB_GREEN    = '#52B92C';
BUSTED_BLUE  = '#5993B5'
class RobbingBanks:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values



    def __init__(self, town_map):
        """ Constructor of the environment town_map.
        """
        self.STEP_REWARD = 0
        self.BANK_REWARD = 10
        self.CAUGHT_REWARD = -50
        self.town_map                 = town_map;
        self.initial_state            = np.array([0,0,1,2])
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards();


    def __actions(self):
        actions = dict();
        actions[self.STAY]       = np.array([0, 0]);
        actions[self.MOVE_LEFT]  = np.array([0,-1]);
        actions[self.MOVE_RIGHT] = np.array([0, 1]);
        actions[self.MOVE_UP]    = np.array([-1,0]);
        actions[self.MOVE_DOWN]  = np.array([1,0]);
        return actions;

    def __states(self):
        states = dict();
        states_vec = dict();

        s = 0;
        for i in range(self.town_map.shape[0]):
            for j in range(self.town_map.shape[1]):
                for k in range(self.town_map.shape[0]):
                    for l in range(self.town_map.shape[1]):
                        states[s] = np.array([i,j,k,l]);
                        states_vec[(i,j,k,l)] = s;
                        s += 1;
                        
        return states, states_vec

    def __move(self, state, action):
        """ Makes a step in the town_map, given a current position and an action.
            If the action STAY or an inadmissible action is used, the robber stays in place.

            :return integer next_cell corresponding to position (x,y) x (x,y) on the town_map that agent transitions to.
        """
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];
        # Is the future position an impossible one ?
        hitting_town_walls =  (row == -1) or (row == self.town_map.shape[0]) or \
                              (col == -1) or (col == self.town_map.shape[1])
        # Based on the impossiblity check return the next state.
        list_police_pos = self.__police_positions(state)
        new_police_pos = list_police_pos[np.random.randint(len(list_police_pos))]
        
        #caught = (row, col) == (new_police_pos[0], new_police_pos[1])
        caught = all(self.states[state][0:2] == self.states[state][2:])
        if caught:
            return self.map[tuple(self.initial_state)];
        #Hot take: If you "unintentionally" hit the wall, the result should be that you (and the police) stay in place since it's not a "deliberate" move
        elif hitting_town_walls:
            return state
        else:
            return self.map[(row, col, new_police_pos[0], new_police_pos[1])];
        
    def __police_positions(self, state):
        """
            Input: The state as an int
            Returns: A list of possible new minotaur positions from current state 
        """
        agent_pos = self.states[state][0:2]
        police_pos = self.states[state][2:]
        diff_pos = np.sign(agent_pos - police_pos)
        list_pos = [[1,0], [-1,0], [0, diff_pos[1]]] if diff_pos[0] == 0 else [[0,1], [0,-1], [diff_pos[0],0]] if diff_pos[1] == 0 else [[0,diff_pos[1]], [diff_pos[0],0]]
        list_pos += police_pos
        list_pos = list(filter(None,[tuple(pos)*(0<=pos[0]<self.town_map.shape[0] and 0<=pos[1]<self.town_map.shape[1]) for pos in list_pos]))
        return list_pos


    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            #if we are in the same position as the police, we return to initial
            if (self.states[s][0],self.states[s][1])==(self.states[s][2],self.states[s][3]):
                transition_probabilities[self.initial_state, s, :] = 1/3

            else:
                for a in range(self.n_actions):
                    list_pos = self.__police_positions(s) #police positions
                    for police_pos in list_pos:
                        next_s = self.__move(s,a);
                        new_pos = np.copy(self.states[next_s])
                        new_pos[2:] = police_pos
                        next_s = self.map[tuple(new_pos)]
                        transition_probabilities[next_s, s, a] = 1/len(list_pos);
        return transition_probabilities;

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions));
        # rewards[i,j,k] = r(s' | s, a): tensor of rewards of dimension S x S x A
        for s in range(self.n_states):
            
            list_pos = self.__police_positions(s)
            for a in range(self.n_actions):
                next_s = self.__move(s,a);  

                #if we can get caught in the next move
                if (tuple(self.states[next_s][0:2]) in list_pos):
                    #if our next position is not a bank
                    if self.town_map[tuple(self.states[next_s][0:2])] != 1:
                        rewards[s,a] = self.CAUGHT_REWARD/len(list_pos)

                    #if our next position is a bank
                    if self.town_map[tuple(self.states[next_s][0:2])] == 1:
                        rewards[s,a] = self.CAUGHT_REWARD/len(list_pos) + (len(list_pos)-1)*self.BANK_REWARD/len(list_pos)

                #if we cannot get caught in the next move
                else:
                    #reward for standing in a bank
                    if self.town_map[tuple(self.states[next_s][0:2])] == 1:
                        rewards[s,a] = self.BANK_REWARD

            

            # list_pos = self.__police_positions(s)
            # for a in range(self.n_actions):
            #     next_s = self.__move(s,a);

        return rewards;

    def simulate(self,policy):
        path = list();

        # Initialize current state, next state and time
        t = 1;
        s = self.map[tuple(self.initial_state)];
        # Add the starting position in the town_map to the path
        path.append(self.initial_state);
        # Move to next state given the policy and the current state
        next_s = self.__move(s,policy[s]);
        # Add the position in the town_map corresponding to the next state
        # to the pygame.freetype.path
        path.append(self.states[next_s]);
        # Loop while state is not the goal state
        T = 40
        while t<T:
            # Update state
            s = next_s;
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s]);
            # Add the position in the town_map corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Update time and state for next iteration
            t +=1;

        return path


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input town_map env           : The town_map environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;
    #tol = 100
    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 2600:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV

        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))
    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;

def draw_town_map(town_map):

    # Map a color to each cell in the town_map
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = town_map.shape;
    colored_town_map = [[col_map[town_map[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the town_map
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The town_map');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = town_map.shape;
    colored_town_map = [[col_map[town_map[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the town_map
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_town_map,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(town_map, path, save_anim = False, until_caught = False, gamma = 0):

    # Map a color to each cell in the town_map
    col_map = {0: WHITE, 1: SEB_GREEN, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the town_map
    rows,cols = town_map.shape;

    # Create figure of the size of the town_map
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation: $\lambda$ = %0.1f' %gamma);
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_town_map = [[col_map[town_map[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the town_map
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_town_map,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    path_robber = [tuple(p)[0:2] for p in path]
    path_police = [tuple(p)[2:] for p in path]
    for i in range(len(path_robber)):


        if i == 0:
            grid.get_celld()[(path_robber[i])].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[(path_robber[i])].get_text().set_text('Robber')

            grid.get_celld()[(path_police[i])].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path_police[i])].get_text().set_text('Police')    
            if save_anim:
                plt.savefig('optimal_policy_'+str(i))
        else:
            if until_caught and path_robber[i] == path_police[i]:
                grid.get_celld()[(path_robber[i-1])].set_facecolor(col_map[town_map[path_robber[i-1]]])
                grid.get_celld()[(path_robber[i-1])].get_text().set_text('')
                grid.get_celld()[(path_police[i-1])].set_facecolor(col_map[town_map[path_police[i-1]]])
                grid.get_celld()[(path_police[i-1])].get_text().set_text('')    
                grid.get_celld()[(path_police[i])].set_facecolor(BUSTED_BLUE)    
                grid.get_celld()[(path_police[i])].get_text().set_text('BUSTED')
                print("BUSTED!!!", gamma)
                if save_anim:    
                    plt.savefig(str(gamma)+'_'+str(i)+'.png')
                break

            if save_anim:
                plt.savefig(str(gamma)+'_'+str(i)+'.png')
            grid.get_celld()[(path_robber[i-1])].set_facecolor(col_map[town_map[path_robber[i-1]]])
            grid.get_celld()[(path_robber[i-1])].get_text().set_text('')
            grid.get_celld()[(path_police[i-1])].set_facecolor(col_map[town_map[path_police[i-1]]])
            grid.get_celld()[(path_police[i-1])].get_text().set_text('') 

            grid.get_celld()[(path_robber[i])].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[(path_robber[i])].get_text().set_text('Robber')

            grid.get_celld()[(path_police[i])].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path_police[i])].get_text().set_text('Police')
        grid.get_celld()[0,0].get_text().set_text('SEB')
        grid.get_celld()[0,0].get_text().set_color('white')
        grid.get_celld()[0,5].get_text().set_text('SEB')
        grid.get_celld()[0,5].get_text().set_color('white')
        grid.get_celld()[2,0].get_text().set_text('SEB')
        grid.get_celld()[2,0].get_text().set_color('white')
        grid.get_celld()[2,5].get_text().set_text('SEB')
        grid.get_celld()[2,5].get_text().set_color('white')
        plt.pause(0.7)
    plt.show()

        
town_map= np.array([
    [ 1, 0, 0, 0, 0, 1],
    [ 0, 0, 0, 0, 0, 0],
    [ 1, 0, 0, 0, 0, 1]
])


rb = RobbingBanks(town_map)
p=rb.transition_probabilities
n=rb.n_states
for s in range(n):
    summ=np.sum(p[:,s,3])
    if summ>1:
        print(rb.states[s])
# PLOTTING VALUE_FUNC(INIT_STATE) AS A FUNCTION OF LAMBDA/GAMMA

"""
gammas = np.linspace(0.01,1,100,endpoint=False)
values = []
for gamma in gammas:
    V, policy = value_iteration(rb, gamma, epsilon = 1e-6)
    values.append(V[rb.map[(0,0,1,2)]])
plt.semilogy(gammas,values,'--')
plt.xlabel('Discount rate $\lambda$')
plt.ylabel('Value function V')
plt.title('Effect of $\lambda$ on V')
plt.plot()
#plt.show()
plt.savefig('Value_2b.png')
"""


# PLOTTING OPTIMAL POLICY FOR DIFFERENT LAMBDAS

"""
gammas = [0.1,0.5,0.8]

for gamma in gammas:
    V, policy = value_iteration(rb, gamma, 1e-6)
    path = rb.simulate(policy)
    animate_solution(town_map, path, save_anim = False, until_caught = True,gamma=gamma)
"""