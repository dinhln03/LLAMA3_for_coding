import numpy as np
import matplotlib.pyplot as plt
import gym
import random

# hyper parameters
# test 1
# alpha = 0.5
# gamma = 0.95
# epsilon = 0.1

epsilon = 0.1
alpha = 0.1
gamma = 0.1

def update_sarsa_table(sarsa, state, action, reward, next_state, next_action, alpha, gamma):
    '''
    update sarsa state-action pair value, main difference from q learning is that it uses epsilon greedy policy
    return action
    '''
    next_max = sarsa[next_state,next_action] # corresponding action-state value to current action

    # print(f'current status is: {type(q[pre_state,action])},{type(alpha)},{type(reward)},{type(gamma)},{type(next_max)}')
    sarsa[state,action] = sarsa[state,action] + alpha * (reward + gamma * next_max - sarsa[state,action])

def epsilon_greedy_policy_sarsa(env, state, sarsa, epsilon):
    '''
    epsilon greedy policy for q learning to generate actions
    '''
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(sarsa[state])

def epsilon_greedy_policy(env, state, q, epsilon):
    '''
    epsilon greedy policy for q learning to generate actions
    '''
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q[state])

def update_q_table(q, pre_state, action, reward, next_state, alpha, gamma):
    '''

    '''
    next_max = np.max(q[next_state]) # max state-action value for next state
    # print(f'current status is: {type(q[pre_state,action])},{type(alpha)},{type(reward)},{type(gamma)},{type(next_max)}')
    q[pre_state,action] = q[pre_state,action] + alpha * (reward + gamma * next_max - q[pre_state,action])


#-----------------------q learning-------------------------------------------
env = gym.make("Taxi-v3")

# initialize q table
q = np.zeros((env.observation_space.n, env.action_space.n))
q_pre = np.zeros((env.observation_space.n, env.action_space.n)) # to check convergence when training
reward_record = []
error_record = []
# loop for each episode:
for episode in range(5000):
    r = 0
    state = env.reset()
    while True:# loop for each step of episode
        # choose A from S using policy derived from Q(e.g, epsilon greedy policy)
        action = epsilon_greedy_policy(env,state,q,epsilon)
        # take action A, observe R, S'
        next_state, reward, done, _ = env.step(action)
        # update Q(S,A)
        update_q_table(q,state,action,reward,next_state,alpha,gamma)
        # S<--S'
        state = next_state
        r += reward
        if done:
            break
    
    reward_record.append(r)
    error = 0
    for i in range(q.shape[0]):
        error = error + np.sum(np.abs(q[i]-q_pre[i]))
        # print(f'{np.abs(q[i]-q_pre[i])},{np.sum(np.abs(q[i]-q_pre[i]))}')
    error_record.append(error)
    q_pre = np.copy(q)

    if episode%100 == 0:
        print(f'{episode}th episode: {r}, {error}')

#close game env
env.close()

#plot diagram
# plt.plot(list(range(5000)),reward_record)
# plt.show()

# plt.plot(list(range(5000)),error_record)
# plt.show()

#double q learning
env = gym.make("Taxi-v3")

# initialize q table
q1 = np.zeros((env.observation_space.n, env.action_space.n))
q2 = np.zeros((env.observation_space.n, env.action_space.n))
q1_pre = np.zeros((env.observation_space.n, env.action_space.n)) # to check convergence when training
q2_pre = np.zeros((env.observation_space.n, env.action_space.n)) # to check convergence when training

# reward and error record
d_reward_record = []
d_error_record = []

# loop for each episode:
for episode in range(5000):
    r = 0
    state = env.reset()
    while True:# loop for each step of episode
        # choose A from S using policy derived from Q1+Q2(e.g, epsilon greedy policy)
        action = epsilon_greedy_policy(env,state,q1+q2,epsilon)
        # take action A, observe R, S'
        next_state, reward, done, _ = env.step(action)
        # with 0.5 probability:
        if random.uniform(0,1) < 0.5:
            update_q_table(q1,state,action,reward,next_state,alpha,gamma)
        else:
            update_q_table(q2,state,action,reward,next_state,alpha,gamma)
        # S<--S'
        state = next_state
        r += reward
        if done:
            break
    
    d_reward_record.append(r)
    error = 0
    for i in range(q.shape[0]):
        error = error + 0.5 * np.sum(np.abs(q1[i]-q1_pre[i])) + 0.5 * np.sum(np.abs(q2[i]-q2_pre[i]))
        # print(f'{np.abs(q[i]-q_pre[i])},{np.sum(np.abs(q[i]-q_pre[i]))}')
    d_error_record.append(error)
    q1_pre = np.copy(q1)
    q2_pre = np.copy(q2)

    if episode%100 == 0:
        print(f'{episode}th episode: {r}, {error}')

#close game env
env.close()

#plot diagram
plt.plot(list(range(5000)),reward_record,label='q learning')
plt.plot(list(range(5000)),d_reward_record,label='double q learning')
plt.legend()
plt.show()

plt.plot(list(range(5000)),error_record,label='q learning')
plt.plot(list(range(5000)),d_error_record, label='double q learning')
plt.legend()
plt.show()
