from flask import Flask, render_template, request
import numpy as np
from joblib import load

app = Flask(__name__)

# Load the trained model
model = load("disaster_prediction_model.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get input data from form
            features = [
                float(request.form['latitude']),
                float(request.form['longitude']),
                float(request.form['avg_temperature']),
                float(request.form['rainfall_mm']),
                float(request.form['humidity_percent']),
                float(request.form['wind_speed_kmph']),
                float(request.form['atmospheric_pressure_hpa']),
                float(request.form['soil_moisture_percent']),
                float(request.form['seismic_activity_richter']),
                float(request.form['elevation_meters']),
                int(request.form['month']),
            ]
            # Reshape for prediction
            input_array = np.array(features).reshape(1, -1)
            predicted_label = model.predict(input_array)[0]
            prediction = f"Predicted Disaster Type: {predicted_label}"
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
