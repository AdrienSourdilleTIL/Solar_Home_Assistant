import gym
from gym import spaces
import numpy as np

class SolarApplianceEnv(gym.Env):
    """
    Custom Gym environment for managing a solar-powered home with a battery and one appliance.
    The RL agent decides when to run the appliance to maximize efficiency.
    """

    def __init__(self):
        super(SolarApplianceEnv, self).__init__()
        
        # Total hours in a day
        self.hours_per_day = 24
        
        # Battery maximum storage capacity (kWh)
        self.battery_capacity = 10.0
        
        # Starting battery level (kWh)
        self.initial_battery = 5.0
        
        # Energy consumed by the appliance when running (kWh)
        self.appliance_consumption = 2.0
        
        # Maximum solar energy production per hour (kWh)
        self.max_solar_output = 5.0
        
        # Action space: 0 = do nothing, 1 = run appliance
        self.action_space = spaces.Discrete(2)
        
        # Observation space:
        # [current hour, current battery level, solar production, appliance already used]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.hours_per_day - 1, self.battery_capacity, self.max_solar_output, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        """
        Resets environment state at the start of each episode.
        """
        self.hour = 0                               # Start at hour 0
        self.battery = self.initial_battery          # Reset battery to initial level
        self.appliance_done = 0                      # Appliance has not been used yet
        return self._get_obs()                       # Return initial observation

    def _get_obs(self):
        """
        Returns the current observation of the environment.
        """
        solar_output = self._solar_production(self.hour)  # Compute solar output for current hour
        return np.array([self.hour, self.battery, solar_output, self.appliance_done], dtype=np.float32)

    def _solar_production(self, hour):
        """
        Simple solar model using a sine wave:
        - Morning and evening: low production
        - Noon: peak production
        - Night: zero production
        """
        return max(0, self.max_solar_output * np.sin((hour - 6) * np.pi / 12))

    def step(self, action):
        """
        Executes one time step in the environment:
        - Updates battery with solar energy
        - Runs appliance if chosen and possible
        - Applies rewards and penalties
        - Advances time by one hour
        """
        solar = self._solar_production(self.hour)                     # Solar energy produced this hour
        self.battery = min(self.battery + solar, self.battery_capacity)  # Charge battery (capped at capacity)
        
        reward = 0  # Default reward
        info = {}   # Extra info for debugging
        
        # If action is "run appliance" and it hasn't been used yet
        if action == 1 and not self.appliance_done:
            if self.battery >= self.appliance_consumption:
                self.battery -= self.appliance_consumption  # Deduct energy
                self.appliance_done = 1                     # Mark appliance as used
                reward += 10                                # Positive reward
            else:
                reward -= 20                                # Penalty for insufficient energy

        self.hour += 1                                      # Advance one hour
        done = self.hour >= self.hours_per_day              # Check if the day is over

        if done and self.appliance_done == 0:               # If day ends without running appliance
            reward -= 10                                    # Apply penalty
        
        obs = self._get_obs()                               # Get new state observation
        return obs, reward, done, info                      # Return step results