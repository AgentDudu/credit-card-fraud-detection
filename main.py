import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocessing import load_and_preprocess_data
from env.fraud_env import FraudDetectionEnv
from agents.linucb import LinUCBAgent 
from agents.neuralucb import NeuralUCBAgent
from agents.supervised import SupervisedAgent
from agents.linepsilon import LinEpsilonAgent
from agents.lints import LinTSAgent
from agents.neuralts import NeuralTSAgent

def calculate_classification_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
        
    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1_score, 4)
    }

import os
import json
import random # NEW
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocessing import load_and_preprocess_data
from env.fraud_env import FraudDetectionEnv
from agents.linucb import LinUCBAgent 
from agents.neuralucb import NeuralUCBAgent
from agents.neuralts import NeuralTSAgent
from agents.supervised import SupervisedAgent
from agents.fed_linucb import FedLinUCBClient, FedLinUCBServer # NEW

def calculate_classification_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
        
    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1_score, 4)
    }

def run_simulation(agent_type, steps=20000):
    csv_path = 'data/raw/creditcard.csv'
    features, labels = load_and_preprocess_data(csv_path)
    env = FraudDetectionEnv(features, labels, dynamic_fn_penalty=True)
    obs, info = env.reset()
    
    n_features = features.shape[1]
    
    is_multi_agent = agent_type in ["isolated_banks", "federated_banks"]
    
    if is_multi_agent:
        clients = [FedLinUCBClient(client_id=i, n_actions=env.action_space.n, n_features=n_features, alpha=0.1) for i in range(3)]
        server = FedLinUCBServer(n_actions=env.action_space.n, n_features=n_features)
        sync_interval = 100

        if agent_type == "linucb":
            agent = LinUCBAgent(n_actions=env.action_space.n, n_features=n_features, alpha=0.1)
        elif agent_type == "neuralucb":
            agent = NeuralUCBAgent(n_actions=env.action_space.n, n_features=n_features, alpha=0.1)
        elif agent_type == "neural_ts":
            agent = NeuralTSAgent(n_actions=env.action_space.n, n_features=n_features, v=0.1)
        elif agent_type in ["logistic", "xgboost"]:
            agent = SupervisedAgent(model_type=agent_type, retrain_interval=1000)

    total_reward = 0
    results_counter = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    reward_history = [] 
    
    print(f"\n--- Running Simulation with {agent_type.upper()} ({steps} steps) ---")
    
    for step in range(steps):
        if is_multi_agent:
            active_bank_idx = random.randint(0, 2)
            active_agent = clients[active_bank_idx]
        else:
            active_agent = agent
            
        action = active_agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        if agent_type in ["logistic", "xgboost"]:
            active_agent.update(action, obs, reward, info)
        else:
            active_agent.update(action, obs, reward)
            
        if agent_type == "federated_banks" and (step + 1) % sync_interval == 0:
            server.aggregate(clients)
            
        total_reward += reward
        results_counter[info['result']] += 1
        reward_history.append(total_reward) 
        obs = next_obs
        
        if terminated:
            break

    tp, tn, fp, fn = results_counter['TP'], results_counter['TN'], results_counter['FP'], results_counter['FN']
    class_metrics = calculate_classification_metrics(tp, tn, fp, fn)
    
    print(f"Total Cumulative Reward: ${total_reward:.2f}")
    print(f"ML Metrics -> Precision: {class_metrics['Precision']} | Recall: {class_metrics['Recall']} | F1: {class_metrics['F1_Score']}")
    
    final_metrics = {"Total_Reward": round(total_reward, 2), **results_counter, **class_metrics}
    return reward_history, final_metrics

if __name__ == "__main__":
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    agents_to_test = ["linucb", "isolated_banks", "federated_banks", "neuralucb"]
    
    histories = {}
    metrics_summary = {}
    
    for agent in agents_to_test:
        r_hist, final_metrics = run_simulation(agent_type=agent, steps=20000)
        histories[agent] = r_hist
        metrics_summary[agent] = final_metrics
        
    print("\n--- Saving Simulation Results ---")
    df_histories = pd.DataFrame(histories)
    df_histories.to_csv(os.path.join(results_dir, "reward_histories.csv"), index=False)
    with open(os.path.join(results_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print("\n--- Generating Visualization ---")
    loaded_df = pd.read_csv(os.path.join(results_dir, "reward_histories.csv"))
    
    plt.figure(figsize=(13, 8))
    colors = {
        "linucb": "purple",
        "isolated_banks": "red",
        "federated_banks": "green",
        "neuralucb": "blue"
    }
    
    for agent in agents_to_test:
        plt.plot(loaded_df[agent], label=agent.upper(), color=colors[agent], linewidth=2)
        
    plt.title("The Value of Federated Learning: Isolated Banks vs Secure Collaboration")
    plt.xlabel("Transactions Processed (Time)")
    plt.ylabel("Cumulative Business Reward ($)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "federated_comparison.png"))
    plt.show()