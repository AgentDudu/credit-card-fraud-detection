from utils.preprocessing import load_and_preprocess_data
from env.fraud_env import FraudDetectionEnv
from agents.linucb import LinUCBAgent 
from agents.neuralucb import NeuralUCBAgent
from agents.supervised import SupervisedAgent # NEW IMPORT

def run_simulation(agent_type="random", steps=20000):
    csv_path = 'data/raw/creditcard.csv'
    features, labels = load_and_preprocess_data(csv_path)
    
    env = FraudDetectionEnv(features, labels, dynamic_fn_penalty=True)
    obs, info = env.reset()
    
    if agent_type == "linucb":
        agent = LinUCBAgent(n_actions=env.action_space.n, n_features=features.shape[1], alpha=0.1)
    elif agent_type == "neuralucb":
        agent = NeuralUCBAgent(n_actions=env.action_space.n, n_features=features.shape[1], alpha=0.1)
    elif agent_type in ["logistic", "random_forest", "xgboost"]:
        agent = SupervisedAgent(model_type=agent_type, retrain_interval=1000)
    
    total_reward = 0
    results_counter = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    
    print(f"\n--- Running Simulation with {agent_type.upper()} Agent ({steps} steps) ---")
    
    for step in range(steps):
        if agent_type == "random":
            action = env.action_space.sample()
        else:
            action = agent.get_action(obs)
            
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        if agent_type in ["linucb", "neuralucb"]:
            agent.update(action, obs, reward)
        elif agent_type in ["logistic", "random_forest", "xgboost"]:
            agent.update(action, obs, reward, info)
            
        total_reward += reward
        results_counter[info['result']] += 1
        obs = next_obs
        
        if (step + 1) % 5000 == 0:
            print(f"Step {step+1}/{steps} | Reward so far: {total_reward:.2f}")

        if terminated:
            break

    print(f"Total Cumulative Reward: {total_reward:.2f}")
    print(f"True Positives (Caught Fraud): {results_counter['TP']}")
    print(f"True Negatives (Allowed Legit): {results_counter['TN']}")
    print(f"False Positives (Friction): {results_counter['FP']}")
    print(f"False Negatives (Missed Fraud): {results_counter['FN']}")

if __name__ == "__main__":
    agents_to_test = ["logistic", "random_forest", "xgboost", "linucb", "neuralucb"]
    
    for agent in agents_to_test:
        run_simulation(agent_type=agent, steps=20000)