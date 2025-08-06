import numpy as np
from envs.solar_appliance_env import SolarApplianceEnv

def evaluate_heuristic_agent(episodes=100):
    env = SolarApplianceEnv()
    total_rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = 1 # always run appliance
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {episodes} episodes (always run appliance): {avg_reward}")
    return avg_reward

if __name__ == "__main__":
    evaluate_heuristic_agent()