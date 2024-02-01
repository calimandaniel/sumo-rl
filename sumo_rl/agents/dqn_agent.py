import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = self.fc2(x)
        return x


class DQNAgent:
    def __init__(self, id, starting_state, state_space, action_space, q_net=None, alpha=0.001, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        self.id = id
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.action = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if q_net is None:
            self.q_network = QNetwork(len(starting_state), action_space.n).to(self.device)
        else:
            self.q_network = q_net.to(self.device)

        #print(state_space.shape)
        #print("starting_state: ", len(starting_state))
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def act(self):
        state_tensor = torch.tensor(self.state, dtype=torch.float32).to(self.device)
        q_values = self.q_network(state_tensor).to(self.device)
        #print("self:state: ", self.state )
        #print("actions: ", self.action_space)
        state_array = (np.array(self.state, dtype=int))
        #print("State_array: ", state_array)
        #print("Qvals: ", q_values)
        self.action = self.exploration.choose(q_values, state_array, self.action_space)
        #print("Returning: ", self.action)
        return self.action

    def learn(self, next_state, reward, done=False):
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)

        q_values = self.q_network(torch.tensor(self.state, dtype=torch.float32).to(self.device))
        next_q_values = self.q_network(next_state_tensor)
        #print("SELF.ACTION: ", self.action)

        target = q_values.clone()
        #target[0][self.action] = reward + self.gamma * torch.max(next_q_values).item() * (1 - done)
        target.view(-1)[self.action] = reward + self.gamma * torch.max(next_q_values).item() * (1 - done)


        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.state = next_state
        self.acc_reward += reward
    
    def eval_step(self, next_state, reward):
        
        self.state = next_state
        self.acc_reward += reward


    
