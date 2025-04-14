import pickle
import numpy as np
from joblib import load

model = load("disaster_prediction_model.pkl")
print("✅ Model loaded successfully.")
print("🔍 Model details:")
print(f"Type of loaded object: {type(model)}")

# Sample input dictionary (11 features)
# Ensure the keys match the order and format you used during training
sample_input = {
    'temperature': 25.2,
    'humidity': 50.5,
    'wind_speed': 2.3,
    'precipitation': 9.0,
    'pressure': 1010,
    # 'region_code': 4,
    'season_code': 2,
    # 'past_disaster_freq': 2,
    'elevation': 45,
    'population_density': 3000,
    'soil_moisture': 1.45
}

# Convert the input to the right format
print(sample_input)
input_array = np.array(list(sample_input.values())).reshape(1, -1)
print(input_array)

# Perform prediction
predicted_label = model.predict(input_array)[0]

print(f"🔮 Predicted Disaster Type: {predicted_label}")
