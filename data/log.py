import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.solar_appliance_env import SolarApplianceEnv

OBS_NAMES = [
    "Hour of Day",
    "Battery Charge",
    "Solar Output",
    "Appliance Running",
    "Month",
    "Intraday Forecast",
    "Day-Ahead Forecast",
    "Weekly Forecast"
]

def format_obs(obs):
    return ", ".join(f"{name}: {val:.2f}" for name, val in zip(OBS_NAMES, obs))

def run_and_log(steps=24):
    env = SolarApplianceEnv()
    obs, _ = env.reset()
    print("🔄 Initial Observation:")
    print("   " + format_obs(obs))

    for step in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f"\n⏱️ Step {step + 1}")
        print(f"   🚀 Action Taken: {action}")
        print(f"   👀 Observation: {format_obs(obs)}")
        print(f"   💰 Reward: {reward:.2f}")
        if info:
            print(f"   ℹ️ Info: {info}")

        if done:
            print("\n✅ Episode finished early")
            break

    env.close()

if __name__ == "__main__":
    run_and_log()
