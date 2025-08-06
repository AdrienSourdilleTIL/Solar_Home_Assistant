import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SolarApplianceEnv(gym.Env):
    """
    Custom Gym environment for managing a solar-powered home with a battery and one main appliance.
    Battery is continuously used to run other essential devices (e.g., fridge, router).
    """
    def __init__(self):
        super(SolarApplianceEnv, self).__init__()
        
        # Total hours in a day
        self.hours_per_day = 24
        
        # Battery maximum storage capacity (kWh)
        self.battery_capacity = 10.0
        
        # Starting battery level (kWh)
        self.initial_battery = 5.0
        
        # Main appliance consumption (kWh)
        self.appliance_consumption = 2.0
        
        # Baseline household consumption (fridge, router...) (kWh/hour)
        self.baseline_consumption = 0.5
        
        # Maximum solar energy production per hour (kWh)
        self.max_solar_output = 5.0
        
        # Action space: 0 = do nothing, 1 = run main appliance
        self.action_space = spaces.Discrete(2)
        
        # Observation space: [hour, battery level, solar production, appliance used?]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.hours_per_day - 1, self.battery_capacity, self.max_solar_output, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        """Resets the environment at the start of each episode."""
        if hasattr(self, 'battery'):
            # Adjust battery based on previous net import/export if you want persistence
            net_energy = self.total_energy_imported - self.total_energy_exported
            self.battery = max(0, min(self.battery - net_energy, self.battery_capacity))
        else:
            self.battery = self.initial_battery
        
        self.hour = 0
        self.appliance_done = 0
        self.total_energy_imported = 0.0
        self.total_energy_exported = 0.0
        
        return self._get_obs()

    def _get_obs(self):
        """Returns the current observation of the environment."""
        solar_output = self._solar_production(self.hour)
        return np.array([self.hour, self.battery, solar_output, self.appliance_done], dtype=np.float32)

    def _solar_production(self, hour):
        """Simple solar model: peak at noon, zero at night."""
        return max(0, self.max_solar_output * np.sin((hour - 6) * np.pi / 12))

    def step(self, action):
        """Executes one time step in the environment."""
        solar = self._solar_production(self.hour)
        
        # Charge battery with solar energy
        self.battery = min(self.battery + solar, self.battery_capacity)
        
        # Baseline household consumption (always drains the battery)
        self.battery = max(0, self.battery - self.baseline_consumption)
        
        reward = 0
        info = {}

        # If agent chooses to run the main appliance
        if action == 1 and not self.appliance_done:
            if self.battery >= self.appliance_consumption:
                self.battery -= self.appliance_consumption
                self.appliance_done = 1
                reward += 10
            else:
                reward -= 20  # Penalty if not enough energy
        
        self.hour += 1
        done = self.hour >= self.hours_per_day

        # blackout penalty
        if self.battery == 0:
            reward -= 100  # High cost for running out of energy

        # Penalty if the agent didn't run the appliance by end of day
        if done and self.appliance_done == 0:
            reward -= 10

        return self._get_obs(), reward, done, info
