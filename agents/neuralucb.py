import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class BanditNetwork(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super(BanditNetwork, self).__init__()
        self.layer1 = nn.Linear(n_features, 64)
        self.layer2 = nn.Linear(64, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        features = torch.relu(self.layer2(x)) 
        reward = self.out(features)
        return reward, features

class NeuralUCBAgent:
    def __init__(self, n_actions, n_features, alpha=0.1, hidden_dim=32, lr=0.001, memory_size=2000, batch_size=32):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        
        self.models = {a: BanditNetwork(n_features, hidden_dim) for a in range(n_actions)}
        self.optimizers = {a: optim.Adam(self.models[a].parameters(), lr=lr) for a in range(n_actions)}
        self.criterion = nn.MSELoss()
        
        self.A = {a: np.identity(hidden_dim) for a in range(n_actions)}
        self.A_inv = {a: np.identity(hidden_dim) for a in range(n_actions)}
        
        self.memory = {a: deque(maxlen=memory_size) for a in range(n_actions)}
        
    def get_action(self, context):
        x = torch.FloatTensor(context).unsqueeze(0) 
        p = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            self.models[a].eval() 
            with torch.no_grad():
                expected_reward, features = self.models[a](x)
                
            features_np = features.numpy().reshape(-1, 1)
            cb = self.alpha * np.sqrt((features_np.T @ self.A_inv[a] @ features_np).item())
            
            p[a] = expected_reward.item() + cb
            
        return np.argmax(p)
        
    def update(self, action, context, reward):
        self.memory[action].append((context, reward))
        
        current_batch_size = min(len(self.memory[action]), self.batch_size)
        batch = random.sample(self.memory[action], current_batch_size)
        
        batch_contexts = torch.FloatTensor(np.array([item[0] for item in batch]))
        batch_rewards = torch.FloatTensor(np.array([[item[1]] for item in batch]))
        
        self.models[action].train()
        self.optimizers[action].zero_grad()
        
        pred_rewards, _ = self.models[action](batch_contexts)
        
        loss = self.criterion(pred_rewards, batch_rewards)
        loss.backward()
        self.optimizers[action].step()
        
        self.models[action].eval()
        with torch.no_grad():
            x = torch.FloatTensor(context).unsqueeze(0)
            _, new_features = self.models[action](x)
            
        f = new_features.numpy().reshape(-1, 1)
        self.A[action] += f @ f.T
        self.A_inv[action] = np.linalg.inv(self.A[action])