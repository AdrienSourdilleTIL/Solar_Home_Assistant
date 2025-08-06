import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.solar_appliance_env import SolarApplianceEnv  # Your custom environment

def evaluate_model(model_path="ppo_solar_agent.zip", episodes=1, plot_episode=1):
    env = SolarApplianceEnv()
    model = PPO.load(model_path) 

    rewards = []
    battery_trace = []  # Store battery levels for visualization
    hours_trace = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        episode_battery = []
        episode_hours = []

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Track battery level for the first plotted episode
            if ep + 1 == plot_episode:
                episode_battery.append(env.battery)
                episode_hours.append(env.current_step)

        rewards.append(total_reward)
        print(f"Episode {ep+1}: Total Reward = {total_reward}")

        # Save the trace of the selected episode
        if ep + 1 == plot_episode:
            battery_trace = episode_battery
            hours_trace = episode_hours

    avg_reward = np.mean(rewards)
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward}")


    # Plot battery level per hour for the selected episode
    plt.figure(figsize=(12, 4))
    plt.plot(hours_trace, battery_trace, color='blue')
    plt.xlabel('Hour')
    plt.ylabel('Battery Level')
    plt.title(f'Battery Level Over Time (Episode {plot_episode})')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    evaluate_model()
