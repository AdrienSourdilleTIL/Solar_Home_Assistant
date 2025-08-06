import pandas as pd
from pathlib import Path

input_file = Path(r"C:\Users\AdrienSourdille\Solar_Home_Assistant\data\ninja_pv_46.1858_-1.4086_corrected.csv")
output_dir = input_file.parent

# Load CSV
df = pd.read_csv(input_file, parse_dates=['local_time'])

# Add hour column but keep local_time for grouping
df['hour'] = df['local_time'].dt.hour

# Group by date and save only hour and production
for date, group in df.groupby(df['local_time'].dt.date):
    folder = output_dir / f"{date.year:04d}" / f"{date.month:02d}" / f"{date.day:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    group[['hour', 'electricity']].rename(columns={'electricity': 'production'}).to_csv(folder / "data.csv", index=False)

print("CSV files created successfully")
