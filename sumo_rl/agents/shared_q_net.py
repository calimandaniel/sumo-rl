import torch
import torch.nn as nn


class SharedQNetwork(nn.Module):
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