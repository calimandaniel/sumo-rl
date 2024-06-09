"""Epsilon Greedy Exploration Strategy."""
import numpy as np
import torch

class EpsilonGreedy:
    """Epsilon Greedy Exploration Strategy."""

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
        """Initialize Epsilon Greedy Exploration Strategy."""
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose_nn(self, q_table, action_space):
        """Choose action based on epsilon greedy strategy."""
        if np.random.rand() < self.epsilon:
            action = int(action_space.sample())
        else:
            action = torch.argmax(q_table).item()

        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        #print(self.epsilon)
        return action
    
    def choose_q_table(self, q_table, state, action_space):
        """Choose action based on epsilon greedy strategy."""
        if np.random.rand() < self.epsilon:
            action = int(action_space.sample())
        else:
            action = np.argmax(q_table[state])

        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        # print(self.epsilon)
        return action
    
    def choose(self, q_table, action_space):
        """Choose action based on epsilon greedy strategy."""
        if np.random.rand() < self.epsilon:
            action = np.random.randint(action_space)  # Choose a random action
        else:
            action = np.argmax(q_table)

        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        return action

    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.initial_epsilon
