import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy

class RandQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = self.fc2(x)
        return x
    
    def save(self, name):
        name = ".\\models\\" + name + ".pt"
        torch.save(self.state_dict(), name)

    def load(self, name):
        name = ".\\models\\" + name + ".pt"
        self.load_state_dict(torch.load(name))

class RandAgent:
    def __init__(self, id, starting_state, state_space, action_space, q_net=None, alpha=0.001, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        self.id = id
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.action = None
        self.exploration = exploration_strategy
        self.acc_reward = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rewards = []
        
    def act(self):
        
        self.action = int(self.action_space.sample())
        return self.action

    def learn(self, next_state, reward, done=False):
        #self.acc_reward += reward
        self.rewards.append(reward)
        return
    
    def eval_step(self, next_state, reward):
        
        self.state = next_state
        self.acc_reward += reward
