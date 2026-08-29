import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from utils.preprocessing import load_and_preprocess_data
from env.fraud_env import FraudDetectionEnv
from agents.linucb import LinUCBAgent 
from agents.neuralucb import NeuralUCBAgent
from agents.supervised import SupervisedAgent
from agents.linepsilon import LinEpsilonAgent
from agents.lints import LinTSAgent

def run_simulation(agent_type="random", steps=20000):
    csv_path = 'data/raw/creditcard.csv'
    features, labels = load_and_preprocess_data(csv_path)
    
    env = FraudDetectionEnv(features, labels, dynamic_fn_penalty=True)
    obs, info = env.reset()
    
    if agent_type == "linucb":
        agent = LinUCBAgent(n_actions=env.action_space.n, n_features=features.shape[1], alpha=0.1)
    elif agent_type == "neuralucb":
        agent = NeuralUCBAgent(n_actions=env.action_space.n, n_features=features.shape[1], alpha=0.1)
    elif agent_type == "lin_epsilon":
        agent = LinEpsilonAgent(n_actions=env.action_space.n, n_features=features.shape[1], epsilon=0.05)
    elif agent_type == "lin_ts":
        agent = LinTSAgent(n_actions=env.action_space.n, n_features=features.shape[1], v=0.1)
    elif agent_type in ["logistic", "random_forest", "xgboost"]:
        agent = SupervisedAgent(model_type=agent_type, retrain_interval=1000)
    
    total_reward = 0
    results_counter = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    reward_history = [] 
    
    print(f"\n--- Running Simulation with {agent_type.upper()} Agent ({steps} steps) ---")
    
    for step in range(steps):
        if agent_type == "random":
            action = env.action_space.sample()
        else:
            action = agent.get_action(obs)
            
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        if agent_type in ["linucb", "neuralucb", "lin_epsilon", "lin_ts"]:
            agent.update(action, obs, reward)
        elif agent_type in ["logistic", "random_forest", "xgboost"]:
            agent.update(action, obs, reward, info)
            
        total_reward += reward
        results_counter[info['result']] += 1
        reward_history.append(total_reward) 
        obs = next_obs
        
        if terminated:
            break

    print(f"Total Cumulative Reward: {total_reward:.2f}")
    print(f"True Positives: {results_counter['TP']} | False Positives: {results_counter['FP']}")
    print(f"True Negatives: {results_counter['TN']} | False Negatives: {results_counter['FN']}")
    
    return reward_history, results_counter, total_reward

if __name__ == "__main__":
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    agents_to_test = ["random_forest", "logistic", "xgboost", "lin_epsilon", "lin_ts", "linucb", "neuralucb"]
    histories = {}
    metrics = {}
    
    for agent in agents_to_test:
        r_hist, counts, final_reward = run_simulation(agent_type=agent, steps=20000)
        
        histories[agent] = r_hist
        metrics[agent] = {
            "Total_Reward": final_reward,
            "TP": counts["TP"],
            "TN": counts["TN"],
            "FP": counts["FP"],
            "FN": counts["FN"]
        }
        
    print("\n--- Saving Simulation Results ---")
    df_histories = pd.DataFrame(histories)
    csv_out_path = os.path.join(results_dir, "reward_histories.csv")
    df_histories.to_csv(csv_out_path, index=False)
    
    json_out_path = os.path.join(results_dir, "metrics_summary.json")
    with open(json_out_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\n--- Generating Visualization from Saved Data ---")
    loaded_df = pd.read_csv(csv_out_path)
    
    plt.figure(figsize=(13, 8))
    
    colors = {
        "random_forest": "red",
        "logistic": "orange",
        "xgboost": "blue",
        "lin_epsilon": "magenta",
        "lin_ts": "cyan",
        "linucb": "purple",
        "neuralucb": "green"
    }
    
    for agent in agents_to_test:
        plt.plot(loaded_df[agent], label=agent.upper(), color=colors[agent], linewidth=2)
        
    plt.title("Cumulative Reward: Online RL (Bandits) vs Batch Supervised Learning")
    plt.xlabel("Transactions Processed (Time)")
    plt.ylabel("Cumulative Business Reward ($)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, "reward_comparison.png")
    plt.savefig(plot_path)
    print(f"[SUCCESS] Saved plot to: {plot_path}")
    
    plt.show()