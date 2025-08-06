import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SolarApplianceEnv(gym.Env):
    """
    Custom Gymnasium environment for managing a solar-powered home with a battery and one main appliance.
    Battery is continuously used to run other essential devices (e.g., fridge, router).
    """

    def __init__(self):
        super(SolarApplianceEnv, self).__init__()
        
        self.hours_per_day = 24
        self.days_per_episode = 7
        self.max_steps = self.hours_per_day * self.days_per_episode  # 168 steps per episode

        self.battery_capacity = 10.0
        self.initial_battery = 10
        self.appliance_consumption = 2.0
        self.baseline_consumption = 0.7
        self.max_solar_output = 5.0

        self.action_space = spaces.Discrete(2)  # 0 = do nothing, 1 = run main appliance

        # Observation: [hour, battery level, solar production, appliance_done]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.hours_per_day - 1, self.battery_capacity, self.max_solar_output, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.battery = self.initial_battery
        self.hour = 0
        self.current_step = 0
        self.appliance_done = 0
        self.total_energy_imported = 0.0
        self.total_energy_exported = 0.0
        self._day_factors = []  # will hold one factor per day
        
        return self._get_obs(), {}

    def _solar_production(self, hour):
        day = self.current_step // self.hours_per_day
        # Generate day factor if not exists
        if len(self._day_factors) <= day:
            self._day_factors.append(np.random.uniform(0.6, 1.0))
        current_day_factor = self._day_factors[day]

        base_output = max(0, self.max_solar_output * np.sin((hour - 6) * np.pi / 12))
        return base_output * current_day_factor

    def _get_obs(self):
        solar_output = self._solar_production(self.hour)
        return np.array([self.hour, self.battery, solar_output, self.appliance_done], dtype=np.float32)

    def step(self, action):
        solar = self._solar_production(self.hour)
        
        # Charge battery with solar energy
        self.battery = min(self.battery + solar, self.battery_capacity)
        
        # Baseline household consumption
        self.battery = max(0, self.battery - self.baseline_consumption)
        
        reward = 0
        info = {}

        if action == 1 and not self.appliance_done:
            if self.battery >= self.appliance_consumption:
                self.battery -= self.appliance_consumption
                self.appliance_done = 1
                # daily positive reward only applied at end of day below
            else:
                reward -= 20

        # Advance time
        self.hour = (self.hour + 1) % self.hours_per_day
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        if self.battery == 0:
            reward -= 100

        # At the end of each day, apply daily rewards/penalties
        if self.hour == 0:  # end of a day (after hour 23)
            if self.appliance_done:
                reward += 20  # reward for running appliance that day
            else:
                reward -= 10  # penalty for not running appliance
            
            self.appliance_done = 0  # reset for next day

        return self._get_obs(), reward, terminated, truncated, info
