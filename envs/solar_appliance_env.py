import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
import random
from datetime import datetime, timedelta

class SolarApplianceEnv(gym.Env):
    def __init__(self, data_dir=r"C:\Users\AdrienSourdille\Solar_Home_Assistant\data"):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.hours_per_day = 24
        self.days_per_episode = 7
        self.max_steps = self.hours_per_day * self.days_per_episode

        self.battery_capacity = 10.0
        self.initial_battery = 5
        self.appliance_consumption = 2.0
        self.baseline_consumption = 0.2
        self.max_solar_output = 5.0

        # Observation space now includes 3 forecast features
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            high=np.array([
                self.hours_per_day - 1, # hour of the day
                self.battery_capacity, # battery charge
                self.max_solar_output, # solar ouptut 
                1, # appliance run
                12, # Month number
                100,   # intraday total forecast (adjust as needed)
                100,   # day-ahead total forecast
                1000   # weekly total forecast
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(2)

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

        def random_date(year=2025):
            start = datetime(year, 1, 1)
            end = datetime(year, 12, 31)
            delta = (end - start).days
            random_days = random.randint(0, delta)
            return start + timedelta(days=random_days)

        def get_episode_dates(data_dir, start_date, days_per_episode=7):
            year = start_date.year
            # Prevent overflow beyond Dec 31
            if start_date + timedelta(days=days_per_episode - 1) > datetime(year, 12, 31):
                start_date = datetime(year, 12, 31) - timedelta(days=days_per_episode - 1)

            episode_dates = []
            for i in range(days_per_episode):
                day = start_date + timedelta(days=i)
                path = Path(data_dir) / str(year) / f"{day.month:02d}" / f"{day.day:02d}" / "data.csv"
                episode_dates.append(path)
            return episode_dates

        start_date = random_date()
        self.start_month = start_date.month
        self._episode_dates = get_episode_dates(self.data_dir, start_date=start_date, days_per_episode=self.days_per_episode)
        self._daily_data_cache = {}

        return self._get_obs(), {}

    def _solar_production(self, hour):
        day_idx = self.current_step // self.hours_per_day
        hour_in_day = hour % self.hours_per_day

        if day_idx >= len(self._episode_dates):
            return 0

        daily_path = self._episode_dates[day_idx]

        if day_idx not in self._daily_data_cache:
            try:
                df = pd.read_csv(daily_path)
                if "hour" not in df.columns:
                    df["hour"] = pd.to_datetime(df["local_time"]).dt.hour
                self._daily_data_cache[day_idx] = df
            except FileNotFoundError:
                print(f"File not found: {daily_path}")
                return 0

        df = self._daily_data_cache[day_idx]
        row = df[df["hour"] == hour_in_day]
        if row.empty:
            return 0

        production = row["production"].iloc[0]
        return min(production, self.max_solar_output)

    def _get_forecasts(self):
        day_idx = self.current_step // self.hours_per_day
        hour_in_day = self.hour % self.hours_per_day
        
        # Intraday forecast (remaining hours today)
        intraday = 0
        if day_idx < len(self._episode_dates):
            df_today = self._daily_data_cache.get(day_idx)
            if df_today is None:
                df_today = pd.read_csv(self._episode_dates[day_idx])
                if "hour" not in df_today.columns:
                    df_today["hour"] = pd.to_datetime(df_today["local_time"]).dt.hour
                self._daily_data_cache[day_idx] = df_today
            intraday = df_today[df_today["hour"] > hour_in_day]["production"].sum()
        
        # Day-ahead forecast (next full day)
        day_ahead = 0
        if day_idx + 1 < len(self._episode_dates):
            df_tomorrow = self._daily_data_cache.get(day_idx + 1)
            if df_tomorrow is None:
                df_tomorrow = pd.read_csv(self._episode_dates[day_idx + 1])
                if "hour" not in df_tomorrow.columns:
                    df_tomorrow["hour"] = pd.to_datetime(df_tomorrow["local_time"]).dt.hour
                self._daily_data_cache[day_idx + 1] = df_tomorrow
            day_ahead = df_tomorrow["production"].sum()
        
        # Weekly forecast (next 7 days)
        weekly = 0
        for i in range(7):
            d_idx = day_idx + i
            if d_idx < len(self._episode_dates):
                df_week = self._daily_data_cache.get(d_idx)
                if df_week is None:
                    df_week = pd.read_csv(self._episode_dates[d_idx])
                    if "hour" not in df_week.columns:
                        df_week["hour"] = pd.to_datetime(df_week["local_time"]).dt.hour
                    self._daily_data_cache[d_idx] = df_week
                weekly += df_week["production"].sum()
        
        # Add noise to forecasts
        intraday *= np.random.normal(1, 0.05)
        day_ahead *= np.random.normal(1, 0.10)
        weekly *= np.random.normal(1, 0.25)
        
        return intraday, day_ahead, weekly

    def _get_obs(self):
        solar_output = self._solar_production(self.hour)
        intraday, day_ahead, weekly = self._get_forecasts()
        
        return np.array([
            self.hour, 
            self.battery, 
            solar_output, 
            self.appliance_done, 
            self.start_month,
            intraday,
            day_ahead,
            weekly
        ], dtype=np.float32)

    def step(self, action):
        solar = self._solar_production(self.hour)

        # Update battery
        self.battery = min(self.battery + solar, self.battery_capacity)
        self.battery = max(0, self.battery - self.baseline_consumption)

        reward = 0
        info = {}

        if action == 1:
            if self.battery >= self.appliance_consumption:
                self.battery -= self.appliance_consumption
                reward += 20  # Successfully ran appliance
            else:
                reward -= 5  # Tried to run appliance but failed

        reward -= 1  # Efficiency penalty

        self.hour = (self.hour + 1) % self.hours_per_day
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        if self.battery == 0:
            reward -= 100  # Blackout penalty

        return self._get_obs(), reward, terminated, truncated, info
