import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from envs.solar_appliance_env import SolarApplianceEnv  # Your custom environment

np.set_printoptions(threshold=np.inf)

def trace_model_behavior(model_path="ppo_solar_agent.zip", episode=1, save_csv=True):
    env = SolarApplianceEnv()
    model = PPO.load(model_path)

    obs, info = env.reset()
    done = False

    logs = {
        "Step": [],
        "Hour": [],
        "Battery": [],
        "SolarOutput": [],
        "Action": [],
        "Reward": [],
        "ApplianceDone": []
    }

    step_count = 0
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

        hour, battery, solar_output, appliance_done = obs

        logs["Step"].append(step_count)
        logs["Hour"].append(hour)
        logs["Battery"].append(battery)
        logs["SolarOutput"].append(solar_output)
        logs["Action"].append(action)
        logs["Reward"].append(reward)
        logs["ApplianceDone"].append(appliance_done)

        total_reward += reward
        step_count += 1

    df = pd.DataFrame(logs)
    print(f"Total reward for episode {episode}: {total_reward}")
    print(df.head(168))  # print first 30 steps

    if save_csv:
        df.to_csv("model_behavior_trace.csv", index=False)
        print("Trace saved to model_behavior_trace.csv")

    return df

if __name__ == "__main__":
    trace_df = trace_model_behavior()
