import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
import random

class SolarApplianceEnv(gym.Env):
    def __init__(self, data_dir=r"C:\Users\AdrienSourdille\Solar_Home_Assistant\data"):
        super(SolarApplianceEnv, self).__init__()
        
        self.data_dir = Path(data_dir)
        self.hours_per_day = 24
        self.days_per_episode = 7
        self.max_steps = self.hours_per_day * self.days_per_episode

        self.battery_capacity = 10.0
        self.initial_battery = 10
        self.appliance_consumption = 2.0
        self.baseline_consumption = 0.7
        self.max_solar_output = 5.0

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.hours_per_day - 1, self.battery_capacity, self.max_solar_output, 1], dtype=np.float32),
            dtype=np.float32
        )

        # Variables for real data
        self._daily_data_cache = {}
        self._episode_dates = []

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.battery = self.initial_battery
        self.hour = 0
        self.current_step = 0
        self.appliance_done = 0
        self.total_energy_imported = 0.0
        self.total_energy_exported = 0.0

        # Pick a random start date from available data
        all_dates = sorted(self.data_dir.rglob("data.csv"))
        start_idx = random.randint(0, len(all_dates) - self.days_per_episode)
        self._episode_dates = all_dates[start_idx:start_idx + self.days_per_episode]

        # Clear cache
        self._daily_data_cache = {}

        return self._get_obs(), {}

    def _solar_production(self, hour):
        day = self.current_step // self.hours_per_day
        daily_path = self._episode_dates[day]

        if day not in self._daily_data_cache:
            df = pd.read_csv(daily_path)
            if "hour" not in df.columns:
                df["hour"] = pd.to_datetime(df["local_time"]).dt.hour
            self._daily_data_cache[day] = df

        df = self._daily_data_cache[day]
        row = df[df["hour"] == hour]
        if row.empty:
            return 0

        # If you want to keep scaling to max_solar_output
        production = row["production"].iloc[0]
        return min(production, self.max_solar_output)

    def _get_obs(self):
        solar_output = self._solar_production(self.hour)
        return np.array([self.hour, self.battery, solar_output, self.appliance_done], dtype=np.float32)

    def step(self, action):
        solar = self._solar_production(self.hour)

        self.battery = min(self.battery + solar, self.battery_capacity)
        self.battery = max(0, self.battery - self.baseline_consumption)

        reward = 0
        info = {}

        if action == 1 and not self.appliance_done:
            if self.battery >= self.appliance_consumption:
                self.battery -= self.appliance_consumption
                self.appliance_done = 1
            else:
                reward -= 20

        self.hour = (self.hour + 1) % self.hours_per_day
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        if self.battery == 0:
            reward -= 100

        if self.hour == 0:
            if self.appliance_done:
                reward += 20
            else:
                reward -= 10
            self.appliance_done = 0

        return self._get_obs(), reward, terminated, truncated, info
