from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        features = [float(x) for x in request.form.values()]
        input_data = np.array([features])
        input_df = pd.DataFrame(input_data)

        # Make prediction
        prediction = model.predict(input_df)
        result = int(prediction[0])

        # Return result message
        if result == 1:
            message = "⚠️ High Risk: The patient is likely to have heart disease."
        else:
            message = "✅ Low Risk: The patient is unlikely to have heart disease."

        return render_template('result.html', prediction_text=message)

    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

