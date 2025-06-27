import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template
import os

app = Flask(__name__)

# Load model, scaler, and preprocessor
model = joblib.load(r"Flask\model.pkl")
scaler = joblib.load(r"Flask\scale.pkl")
preprocessor = joblib.load(r"Flask\encoder.pkl")  # Load the ColumnTransformer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data = {
            'holiday': request.form['holiday'],
            'temp': float(request.form['temp']),
            'rain': float(request.form['rain']),
            'snow': float(request.form['snow']),
            'weather': request.form['weather'],
            'day': request.form['day'],
            'month': request.form['month'],
            'year': request.form['year'],
            'hours': request.form['hours'],
            'minutes': request.form['minutes'],
            'seconds': request.form['seconds']
        }
        df = pd.DataFrame([data])

        # Preprocess the data using the loaded preprocessor and scaler
        x_processed = preprocessor.transform(df)
        x_scaled = scaler.transform(x_processed)

        # Predict
        prediction = model.predict(x_scaled)
        return render_template('chance.html', prediction=prediction[0])
    except Exception as e:
        return render_template('noChance.html', error=str(e))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)