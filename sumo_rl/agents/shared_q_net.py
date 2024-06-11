import torch
import torch.nn as nn


class SharedQNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=3, neurons=128, use_ln=False, use_dropout=False, dropout_rate=0.2):
        super(SharedQNetwork, self).__init__()
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