import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class PPOCritic(nn.Module):
    def __init__(self, input_d, hidden_d, hidden_act=nn.ReLU, lr=1e-4):
        super(PPOCritic, self).__init__()
        
        # Build the shared MLP body
        layers = []
        prev_d = input_d
        for h_d in hidden_d:
            layers.append(nn.Linear(prev_d, h_d))
            layers.append(hidden_act())
            prev_d = h_d
        
        self.model_body = nn.Sequential(*layers)

        self.delay_head = nn.Linear(prev_d, 1)
        self.jsma_head = nn.Linear(prev_d, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        # Pass the state through the shared body
        shared_features = self.model_body(state)
        
        # Get values from both heads ---
        value_delay = self.delay_head(shared_features)
        value_jsma = self.jsma_head(shared_features)
        
        # Return both values
        return value_delay, value_jsma

    def get_weights(self, nettype=None):
        return self.state_dict()

    def set_weights(self, weights, nettype=None):
        self.load_state_dict(weights)

    def save_weights(self, nettype, path, fname):
        if not fname.endswith('.pt'):
            fname += '.pt'
        filepath = os.path.join(path, fname)
        torch.save(self.state_dict(), filepath)

    def load_weights(self, path):
        if not path.endswith('.pt'):
            path += '.pt'
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            print(f"Successfully loaded weights from {path}")
        else:
            print(f"Warning: Weight file not found at {path}")