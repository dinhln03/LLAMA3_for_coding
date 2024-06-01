import numpy as np

class Reward:
	pass

class StaticReward(Reward):
	def __init__(self, value):
		self.value = value

	def get(self):
		return value

class NormalReward(Reward):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def get(self):
		return np.random.normal(self.mean, self.std)

class Bandit:
	def __init__(self, arms):
		self.no_of_arms = arms
		self.arms = [np.random.normal(0, 1) for _ in range(arms)]

	def step(self, arm):
		return np.random.normal(self.arms[arm], 1)

class MDP:
	"""
	Represents a Markov Decision Process.
	"""
	def __init__(self, S, A, R, p):
		"""
		Parameters
		----------
		S : int
			Number of states
		A : matrix
			A[s][a] is True iff a is permitted in s
		R : list
			A list of reward generators
		p : matrix
			p[s][a][s'] = p(s'|s,a)
		"""

		self.S = list(range(S))
		self.A, self.R, self.p = A, R, p
		self.no_of_states = S
		self.no_of_actions = len(A[0])

	def step(self, s, a):
		"""Given a state and an action, returns a new state and a reward.

		Parameters
		----------
		s : int
			Current state
		a : int
			Action to take
		"""

		s_prime = np.random.choice(self.no_of_states, p = self.p[s][a])
		r = self.R[s_prime].get()

		return s_prime, r

def epsilon_greedy(no_of_arms, epsilon, Q, N):
	if np.random.random() > epsilon:
		# greedy
		action = np.argmax(Q)
	else:
		# random
		action = np.random.choice(no_of_arms)

	return action

def main():
	no_of_arms = 10
	no_of_steps = 1000
	epsilon = 0.1

	no_of_runs = 2000

	#bandit = Bandit(no_of_arms)

	arms = np.random.normal(0, 1, no_of_arms)

	S = 1
	A = [[True] * no_of_arms]
	R = [NormalReward(m, 1) for m in arms]
	p = [[[1] for _ in range(no_of_arms)]]

	bandit = MDP(S, A, R, p)
	
	#optimal_action = np.argmax(bandit.arms)
	optimal_action = np.argmax(arms)

	np.random.seed(1)

	Q = [[0] * no_of_arms] * no_of_runs
	N = [[0] * no_of_arms] * no_of_runs

	mean_rewards = [0] * no_of_steps

	for j in range(no_of_steps):
		for i in range(no_of_runs):
			action = epsilon_greedy(no_of_arms, epsilon, Q[i], N[i])

			#reward = bandit.step(action)
			_, reward = bandit.step(0, action)

			mean_rewards[j] += reward

			N[i][action] += 1
			Q[i][action] += (1 / N[i][action]) * (reward - Q[i][action])

		mean_rewards[j] /= no_of_runs

if __name__ == '__main__':
	main()