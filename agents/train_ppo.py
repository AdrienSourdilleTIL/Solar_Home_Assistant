from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.solar_appliance_env import SolarApplianceEnv

def main():
    # 1. Initialize environment
    env = SolarApplianceEnv()

    # 2. Sanity check
    check_env(env)

    # 3. Initialize PPO agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./ppo_solar_tensorboard/"
    )

    # 4. Train agent
    model.learn(total_timesteps=100_000)

    # 5. Save model
    model.save("ppo_solar_agent")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
