import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FraudDetectionEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, features, labels, dynamic_fn_penalty=False):
        super(FraudDetectionEnv, self).__init__()
        
        self.features = features
        self.labels = labels
        self.n_samples = len(features)
        self.current_step = 0
        
        self.dynamic_fn_penalty = dynamic_fn_penalty
        
        # Action Space: 0 = Approve, 1 = Decline
        self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.features.shape[1],), dtype=np.float32
        )
        
        self.reward_dict = {
            'TP': 10.0,   # True Positive: Correctly blocked fraud
            'TN': 1.0,    # True Negative: Correctly approved legitimate
            'FP': -2.0,   # False Positive: Wrongly blocked legitimate (Customer friction)
            'FN': -50.0   # False Negative: Wrongly approved fraud (Financial loss)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.features[self.current_step]
        
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):        
        true_label = self.labels[self.current_step]
        obs = self.features[self.current_step]
        transaction_amount = obs[0]
        
        reward = 0.0
        info = {}
        
        if action == 1 and true_label == 1:
            reward = self.reward_dict['TP']
            info['result'] = 'TP'
        elif action == 0 and true_label == 0:
            reward = self.reward_dict['TN']
            info['result'] = 'TN'
        elif action == 1 and true_label == 0:
            reward = self.reward_dict['FP']
            info['result'] = 'FP'
        elif action == 0 and true_label == 1:
            if self.dynamic_fn_penalty:
                reward = self.reward_dict['FN'] * max(1.0, transaction_amount)
            else:
                reward = self.reward_dict['FN']
            info['result'] = 'FN'
            
        self.current_step += 1
        
        terminated = self.current_step >= self.n_samples
        truncated = False
        
        next_obs = None
        if not terminated:
            next_obs = self.features[self.current_step]
        else:
            next_obs = self.features[self.current_step - 1]

        return np.array(next_obs, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step}/{self.n_samples}")