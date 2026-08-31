import numpy as np

class FedLinUCBClient:
    def __init__(self, client_id, n_actions, n_features, alpha=0.1):
        """A Local Bank Agent that learns only from its own customers."""
        self.client_id = client_id
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
        
    def get_params(self):
        return self.A, self.b
        
    def set_params(self, global_A, global_b):
        self.A = {a: np.copy(global_A[a]) for a in range(self.n_actions)}
        self.b = {a: np.copy(global_b[a]) for a in range(self.n_actions)}


class FedLinUCBServer:
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features
        self.global_A = {a: np.identity(n_features) for a in range(n_actions)}
        self.global_b = {a: np.zeros((n_features, 1)) for a in range(n_actions)}

    def aggregate(self, clients):
        new_A = {a: np.identity(self.n_features) for a in range(self.n_actions)}
        new_b = {a: np.zeros((self.n_features, 1)) for a in range(self.n_actions)}
        
        for client in clients:
            client_A, client_b = client.get_params()
            for a in range(self.n_actions):
                new_A[a] += (client_A[a] - np.identity(self.n_features))
                new_b[a] += client_b[a]
                
        self.global_A = new_A
        self.global_b = new_b
        
        for client in clients:
            client.set_params(self.global_A, self.global_b)