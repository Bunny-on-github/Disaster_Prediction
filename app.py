from flask import Flask, render_template, request
import requests
import numpy as np
from joblib import load
from geopy.geocoders import Nominatim
from datetime import datetime
import time

app = Flask(__name__)
model = load("disaster_prediction_model.pkl")
label_encoder = load("label_encoder.pkl")  # Load label encoder to decode predicted labels

# OpenWeatherMap API setup
WEATHER_API_KEY = "1dbfbfbe65c8c71972db914b3a8ce8de"
WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"

# Function to get weather data by city
def get_weather_data(city):
    params = {
        'q': city,
        'appid': WEATHER_API_KEY,
        'units': 'metric'
    }
    try:
        response = requests.get(WEATHER_URL, params=params)
        data = response.json()

        if response.status_code != 200 or 'main' not in data:
            return None

        weather = {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed'],
            'rainfall': data.get('rain', {}).get('1h', 0.0),
            'latitude': data['coord']['lat'],
            'longitude': data['coord']['lon']
        }
        return weather
    except Exception as e:
        print(f"Weather API error: {e}")
        return None

# Function to get location coordinates using city and country
def get_location_data(city, country, retries=3):
    geo = Nominatim(user_agent="disaster-predictor", timeout=5)
    for attempt in range(retries):
        try:
            location = geo.geocode(f"{city}, {country}")
            if location:
                return location.latitude, location.longitude
        except Exception as e:
            print(f"Retry {attempt+1}/{retries} - Geolocation error: {e}")
            time.sleep(1 + attempt)
    return 0.0, 0.0

# Generate features for the model input
def generate_features(lat, lon, weather):
    return [
        lat,                              # latitude
        lon,                              # longitude
        weather.get('temperature', 28),   # avg_temperature
        weather.get('rainfall', 150),     # rainfall_mm
        weather.get('humidity', 65),      # humidity_percent
        weather.get('wind_speed', 12),    # wind_speed_kmph
        weather.get('pressure', 1012),    # atmospheric_pressure_hpa
        45.0,                             # soil_moisture_percent
        2.5,                              # seismic_activity_richter (default)
        300,                              # elevation_meters (default)
        datetime.now().month              # month
    ]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    weather_data = None
    error = None
    input_data = []
    probabilities = []

    if request.method == "POST":
        city = request.form.get("city")
        country = request.form.get("country")
        # disaster_type = request.form.get("disaster")

        weather_data = get_weather_data(city)
        if not weather_data:
            error = "Unable to retrieve weather data. Please verify the city name."
            return render_template("index.html", prediction=None, error=error)

        lat, lon = get_location_data(city, country)
        features = generate_features(lat, lon, weather_data)
        input_array = np.array(features).reshape(1, -1)
        
        # Get the probabilities for all disaster types
        disaster_probabilities = model.predict_proba(input_array)[0]
        probabilities = dict(zip(label_encoder.classes_, disaster_probabilities))  # Map disaster types to probabilities
        print("Probabilities:", probabilities)
        
        # Get the predicted label (disaster with highest probability)
        predicted_label = label_encoder.inverse_transform([np.argmax(disaster_probabilities)])[0]
        prediction = predicted_label
        input_data = features

    return render_template("index.html", prediction=prediction, weather=weather_data, error=error, 
                           input_data=input_data, probabilities=probabilities)

if __name__ == "__main__":
    app.run(debug=True)
    # Set debug=False for production use