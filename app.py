from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import base64
from io import BytesIO
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


app = Flask(__name__)

# Load model and threshold
model = joblib.load('fraud_model.pkl')
threshold = np.load('optimal_threshold.npy')

# SHAP Explainer setup (only works well with tree-based models)
explainer = shap.Explainer(model)

def generate_shap_plot(instance):
    try:
        shap_values = explainer(instance)
        expected_value = explainer.expected_value
        force_plot_html = shap.plots.force(expected_value, shap_values[0], matplotlib=False)
        return force_plot_html
    except Exception as e:
        print(f"SHAP plot generation failed: {e}")
        return None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        # Collect input data from form
        input_data = {
            'Time': float(request.form['Time']),
            'Amount': float(request.form['Amount']),
        }
        for i in range(1, 29):
            input_data[f'V{i}'] = float(request.form[f'V{i}'])

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Add engineered features
        df['hour_of_day'] = (df['Time'] // 3600 % 24).astype(int)
        df['amount_time_ratio'] = df['Amount'] / (df['Time'] + 1)

        # Check feature count
        expected_features = 32
        if df.shape[1] != expected_features:
            return render_template('index.html', prediction_text="Error: Model expects 32 features", show_shap=False)

        # Make prediction
        proba = model.predict_proba(df)[0][1]
        prediction = "FRAUD" if proba >= threshold else "Legitimate Transaction"
        prediction_text = f"Transaction is {prediction} (Probability: {proba:.2f})"

        # Generate SHAP explanation for this instance
        shap_plot_html = generate_shap_plot(df)
        return render_template('index.html', prediction_text=prediction_text, shap_plot=shap_plot_html, show_shap=True)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error during prediction: {e}", show_shap=False)

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        file = request.files['csv_file']
        df = pd.read_csv(file)

        # Validate input columns
        required_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        if not all(col in df.columns for col in required_columns):
            return render_template('index.html', batch_results="Error: CSV missing required columns", show_shap=False)

        # Feature engineering
        df['hour_of_day'] = (df['Time'] // 3600 % 24).astype(int)
        df['amount_time_ratio'] = df['Amount'] / (df['Time'] + 1)

        # Make predictions
        probs = model.predict_proba(df)[:, 1]
        preds = (probs >= threshold).astype(int)
        df['Prediction'] = preds
        df['Probability'] = probs
        df['Label'] = df['Prediction'].apply(lambda x: 'FRAUD' if x else 'Legitimate')

        # Generate HTML table to display
        result_table = df[['Time', 'Amount', 'Probability', 'Label']].to_html(classes='table table-striped', index=False)

        return render_template('index.html', batch_results=result_table, show_shap=False)

    except Exception as e:
        return render_template('index.html', batch_results=f"Error during batch prediction: {e}", show_shap=False)

if __name__ == '__main__':
    app.run(debug=True)