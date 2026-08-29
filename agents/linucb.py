import numpy as np

class LinUCBAgent:
    def __init__(self, n_actions, n_features, alpha=0.1):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha
        
        self.A = {a: np.identity(n_features) for a in range(n_actions)}
        self.b = {a: np.zeros((n_features, 1)) for a in range(n_actions)}
        
    def get_action(self, context):
        x = context.reshape(-1, 1)
        p = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            
            theta = A_inv @ self.b[a]
            
            expected_reward = (theta.T @ x).item()
            
            cb = self.alpha * np.sqrt((x.T @ A_inv @ x).item())
            
            p[a] = expected_reward + cb
            
        return np.argmax(p)
        
    def update(self, action, context, reward):
        x = context.reshape(-1, 1)
        
        self.A[action] += x @ x.T
        self.b[action] += reward * x