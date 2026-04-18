"""
File: rl_agent.py
Author: Vandana B.S
Description:
Implements a Q-learning based reinforcement learning agent for
adaptive preprocessing selection.
"""

import numpy as np


class PreprocessingRLAgent:

    def __init__(self, n_actions=5, learning_rate=0.1, epsilon=0.2):

        self.q_table = {}
        self.n_actions = n_actions
        self.lr = learning_rate
        self.epsilon = epsilon


    def get_state_key(self, features):
        return tuple((np.array(features) * 10).astype(int))


    def select_action(self, state_features):

        key = self.get_state_key(state_features)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        return int(np.argmax(self.q_table[key]))


    def update_q(self, state_features, action, reward):

        key = self.get_state_key(state_features)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)

        gamma = 0.9

        self.q_table[key][action] += self.lr * (
            reward + gamma * np.max(self.q_table[key]) - self.q_table[key][action]
        )
