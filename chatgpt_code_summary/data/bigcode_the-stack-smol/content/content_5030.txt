import random
import math
import numpy as np
from typing import List


class EpsilonGreedy:
    def __init__(self, epsilon: float, counts: List[int], values: List[float]):
        assert epsilon is None or 0.0 <= epsilon <= 1.0
        self.epsilon = epsilon
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms

    def select_arm(self):
        epsilon = self.epsilon
        if epsilon is None:
            t = sum(self.counts) + 1
            epsilon = 1 / math.log(t + 0.0000001)

        # 活用
        if random.random() > epsilon:
            return np.argmax(self.values)
        # 探索
        else:
            return random.randrange(len(self.values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / float(n)) * value + (1 / float(n)) * reward    # online average

    def __str__(self):
        return "EpsilonGreedy(epsilon={0})".format(self.epsilon)
