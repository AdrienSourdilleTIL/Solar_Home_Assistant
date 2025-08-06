import numpy as np
from envs.solar_appliance_env import SolarApplianceEnv

def evaluate_random_policy(episodes=100):
    env = SolarApplianceEnv()
    total_rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = np.random.choice([0, 1])  # 50% chance for 0 or 1
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {episodes} episodes (random policy): {avg_reward}")
    return avg_reward

if __name__ == "__main__":
    evaluate_random_policy()
