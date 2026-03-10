"""
File: rl_agent.py
Author: Vandan V.S
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
        """
        Convert feature vector to discrete state representation.
        """
        return tuple((features * 10).astype(int))


    def select_action(self, state_features):
        """
        Select action using epsilon-greedy strategy.
        """

        key = self.get_state_key(state_features)

        if np.random.rand() < self.epsilon or key not in self.q_table:
            return np.random.randint(self.n_actions)

        return int(np.argmax(self.q_table[key]))


    def update_q(self, state_features, action, reward):
        """
        Update Q-table values.
        """

        key = self.get_state_key(state_features)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)

        self.q_table[key][action] += self.lr * (reward - self.q_table[key][action])