import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=3, neurons=128, use_ln=False, use_dropout=False, dropout_rate=0.2):
        super(QNetwork, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, neurons))
        if use_ln:
            self.layers.append(nn.LayerNorm(neurons))
        if use_dropout:
            self.layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(hidden_layers):
            if i < hidden_layers // 2:
                # Increase the number of neurons in the first half of the hidden layers
                self.layers.append(nn.Linear(neurons, neurons * 2))
                neurons *= 2
            else:
                # Decrease the number of neurons in the second half of the hidden layers
                self.layers.append(nn.Linear(neurons, neurons // 2))
                neurons //= 2

            if use_ln:
                self.layers.append(nn.LayerNorm(neurons))
            if use_dropout:
                self.layers.append(nn.Dropout(dropout_rate))

        # Output layer
        self.layers.append(nn.Linear(neurons, output_size))
        
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def save(self, name):
        name = ".\\models\\" + name + ".pt"
        torch.save(self.state_dict(), name)

    def load(self, name):
        name = ".\\models\\" + name + ".pt"
        self.load_state_dict(torch.load(name))


class DQNAgentNoComm:
    def __init__(self, id, starting_state, state_space, action_space, q_net=None, alpha=0.001, gamma=0.95, exploration_strategy=EpsilonGreedy(), hidden_layers=3, neurons=128):
        self.id = id
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.action = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if q_net is None:
            self.q_network = QNetwork(len(starting_state), action_space.n, hidden_layers, neurons).to(self.device)
        else:
            self.q_network = q_net.to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.exploration = exploration_strategy
        self.acc_reward = 0
        self.rewards = []
        
        
    def act(self):
        state_tensor = torch.tensor(self.state, dtype=torch.float32).to(self.device)
        q_values = self.q_network(state_tensor).to(self.device)
        state_array = (np.array(self.state, dtype=int))
        self.action = self.exploration.choose_nn(q_values, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False):
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)

        q_values = self.q_network(torch.tensor(self.state, dtype=torch.float32).to(self.device))
        next_q_values = self.q_network(next_state_tensor)
        target = q_values.clone()
        target.view(-1)[self.action] = reward + self.gamma * torch.max(next_q_values).item() * (1 - done)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.state = next_state
        self.acc_reward += reward
        self.rewards.append(self.acc_reward)
        
    
    def eval_step(self, next_state, reward):
        
        self.state = next_state
        self.acc_reward += reward

    def save(self, name):
        name = ".\\models\\" + name + "_agent_" + str(self.id)
        torch.save(self.q_network.state_dict(), name + ".pt")

    def load(self, name):
        name = ".\\models\\" + name + "_agent_" + str(self.id) + ".pt"
        self.q_network.load_state_dict(torch.load(name))