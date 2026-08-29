import numpy as np

class LinTSAgent:
    def __init__(self, n_actions, n_features, v=0.1):
        self.n_actions = n_actions
        self.n_features = n_features
        self.v = v
        
        self.A = {a: np.identity(n_features) for a in range(n_actions)}
        self.b = {a: np.zeros((n_features, 1)) for a in range(n_actions)}
        
    def get_action(self, context):
        x = context.reshape(-1, 1)
        p = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            
            theta_hat = (A_inv @ self.b[a]).flatten()
            
            cov = (self.v ** 2) * A_inv
            cov = (cov + cov.T) / 2.0 
            
            theta_sample = np.random.multivariate_normal(theta_hat, cov)
            
            p[a] = np.dot(theta_sample, context)
            
        return np.argmax(p)
        
    def update(self, action, context, reward):
        x = context.reshape(-1, 1)
        self.A[action] += x @ x.T
        self.b[action] += reward * x