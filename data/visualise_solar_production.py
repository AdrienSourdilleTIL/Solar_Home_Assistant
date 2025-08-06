import numpy as np
import matplotlib.pyplot as plt

hours_per_day = 24
days = 7
max_solar_output = 5.0

# Generate a random weather factor for each day
day_factors = [np.random.uniform(0.6, 1.0) for _ in range(days)]

def solar_production(hour, day_factors):
    day = hour // hours_per_day
    factor = day_factors[day]
    base_output = max(0, max_solar_output * np.sin((hour % hours_per_day - 6) * np.pi / 12))
    return base_output * factor

# Compute production for each hour
hours = np.arange(0, hours_per_day * days)
production = [solar_production(h, day_factors) for h in hours]

# Plot
plt.figure(figsize=(12, 5))
plt.plot(hours, production, label="Solar Output (kWh)")
plt.xlabel("Hour")
plt.ylabel("Solar Production (kWh)")
plt.title("Solar Production Over 7 Days")
plt.grid(True)
plt.legend()
plt.show()
