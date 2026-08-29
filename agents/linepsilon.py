import numpy as np

class LinEpsilonAgent:
    def __init__(self, n_actions, n_features, epsilon=0.1):
        self.n_actions = n_actions
        self.n_features = n_features
        self.epsilon = epsilon
        
        self.A = {a: np.identity(n_features) for a in range(n_actions)}
        self.b = {a: np.zeros((n_features, 1)) for a in range(n_actions)}
        
    def get_action(self, context):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
            
        x = context.reshape(-1, 1)
        p = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            expected_reward = (theta.T @ x).item()
            p[a] = expected_reward
            
        return np.argmax(p)
        
    def update(self, action, context, reward):
        x = context.reshape(-1, 1)
        self.A[action] += x @ x.T
        self.b[action] += reward * x