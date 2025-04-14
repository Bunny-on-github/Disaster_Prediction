import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from joblib import dump

# Load your dataset
df = pd.read_csv("synthetic_natural_disasters.csv")

# Encode the disaster_type labels
label_encoder = LabelEncoder()
df['disaster_type_encoded'] = label_encoder.fit_transform(df['disaster_type'])

# Save the label encoder to decode prediction probabilities later
dump(label_encoder, "label_encoder.pkl")

# Features and Target
features = [
    'latitude', 'longitude', 'avg_temperature', 'rainfall_mm', 'humidity_percent',
    'wind_speed_kmph', 'atmospheric_pressure_hpa', 'soil_moisture_percent',
    'seismic_activity_richter', 'elevation_meters', 'month'
]
X = df[features]
y = df['disaster_type_encoded']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier (Random Forest works well for multi-class)
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the model
dump(clf, "disaster_prediction_model.pkl")
