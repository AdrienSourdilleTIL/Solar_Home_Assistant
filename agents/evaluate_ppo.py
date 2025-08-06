import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.solar_appliance_env import SolarApplianceEnv  # Your custom environment

def evaluate_model(model_path="ppo_solar_agent.zip", episodes=100):
    env = SolarApplianceEnv()
    model = PPO.load(model_path)
    
    rewards = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        
        rewards.append(total_reward)
        print(f"Episode {ep+1}: Total Reward = {total_reward}")

    avg_reward = np.mean(rewards)
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward}")

    # Plot rewards
    plt.figure()
    plt.plot(range(1, episodes+1), rewards, marker='o')
    plt.axhline(avg_reward, color='red', linestyle='--', label=f'Average = {avg_reward:.2f}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Model Evaluation')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    evaluate_model()
