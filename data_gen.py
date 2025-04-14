import pandas as pd
import numpy as np
import random

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Disaster types
disasters = [
    "No Disaster", "Flood", "Earthquake", "Hurricane", "Drought",
    "Landslide", "Wildfire", "Tsunami", "Volcanic Eruption", "Heatwave", "Blizzard"
]

# Feature ranges for each disaster
feature_ranges = {
    "No Disaster":                 {"temp": (10, 25), "rain": (0, 20),  "humid": (30, 60), "wind": (0, 20), "press": (990, 1020), "soil": (30, 60), "seismic": (0.0, 2.0), "elev": (0, 1000)},
    "Flood":                      {"temp": (15, 30), "rain": (200, 500), "humid": (70, 100), "wind": (10, 40), "press": (980, 1010), "soil": (60, 100), "seismic": (0.0, 2.5), "elev": (0, 100)},
    "Earthquake":                 {"temp": (10, 30), "rain": (0, 50), "humid": (20, 80), "wind": (0, 20), "press": (990, 1025), "soil": (30, 70), "seismic": (5.5, 9.5), "elev": (0, 2000)},
    "Hurricane":                  {"temp": (25, 35), "rain": (150, 300), "humid": (75, 100), "wind": (100, 250), "press": (930, 980), "soil": (50, 90), "seismic": (0.0, 2.0), "elev": (0, 100)},
    "Drought":                    {"temp": (30, 45), "rain": (0, 10), "humid": (5, 30), "wind": (5, 25), "press": (1000, 1025), "soil": (0, 20), "seismic": (0.0, 1.5), "elev": (0, 500)},
    "Landslide":                  {"temp": (10, 25), "rain": (150, 400), "humid": (70, 100), "wind": (0, 30), "press": (970, 1015), "soil": (70, 100), "seismic": (2.0, 6.0), "elev": (1000, 3000)},
    "Wildfire":                   {"temp": (35, 50), "rain": (0, 10), "humid": (5, 20), "wind": (10, 60), "press": (990, 1020), "soil": (0, 30), "seismic": (0.0, 2.0), "elev": (200, 1500)},
    "Tsunami":                    {"temp": (20, 30), "rain": (0, 30), "humid": (60, 90), "wind": (0, 20), "press": (990, 1020), "soil": (20, 70), "seismic": (7.0, 9.5), "elev": (0, 50)},
    "Volcanic Eruption":          {"temp": (5, 25), "rain": (0, 50), "humid": (20, 80), "wind": (0, 40), "press": (950, 1020), "soil": (40, 80), "seismic": (4.0, 9.0), "elev": (1000, 4000)},
    "Heatwave":                   {"temp": (40, 50), "rain": (0, 2), "humid": (5, 25), "wind": (0, 15), "press": (1010, 1025), "soil": (0, 20), "seismic": (0.0, 1.0), "elev": (0, 800)},
    "Blizzard":                   {"temp": (-30, 0), "rain": (0, 10), "humid": (60, 100), "wind": (20, 80), "press": (980, 1015), "soil": (60, 100), "seismic": (0.0, 1.5), "elev": (100, 2000)},
}

def generate_sample(disaster_type):
    f = feature_ranges[disaster_type]
    return {
        "disaster_type": disaster_type,
        "latitude": round(np.random.uniform(-90, 90), 4),
        "longitude": round(np.random.uniform(-180, 180), 4),
        "avg_temperature": round(np.random.uniform(*f["temp"]), 1),
        "rainfall_mm": round(np.random.uniform(*f["rain"]), 1),
        "humidity_percent": round(np.random.uniform(*f["humid"]), 1),
        "wind_speed_kmph": round(np.random.uniform(*f["wind"]), 1),
        "atmospheric_pressure_hpa": round(np.random.uniform(*f["press"]), 1),
        "soil_moisture_percent": round(np.random.uniform(*f["soil"]), 1),
        "seismic_activity_richter": round(np.random.uniform(*f["seismic"]), 1),
        "elevation_meters": round(np.random.uniform(*f["elev"]), 1),
        "month": np.random.randint(1, 13)
    }

def generate_dataset(num_rows=1000):
    data = []
    for _ in range(num_rows):
        disaster_type = random.choices(disasters, weights=[30,10,10,8,8,6,6,5,5,6,6])[0]  # Slight class imbalance
        data.append(generate_sample(disaster_type))
    return pd.DataFrame(data)

# Example usage
df = generate_dataset(5000)
df.to_csv("synthetic_natural_disasters.csv", index=False)
print("✅ Dataset generated and saved as 'synthetic_natural_disasters.csv'")
