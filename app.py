import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from lstm_model import PowerConsumptionLSTM
import plotly.graph_objects as go
import plotly.express as px
import json

app = Flask(__name__)

# Global variables
df = None
lstm_model = None
model_loaded = False

def load_data():
    """Load the data if not already loaded"""
    global df
    if df is None:
        try:
            df = pd.read_csv('powerconsumption.csv')
            df['total_power'] = df['PowerConsumption_Zone1'] + df['PowerConsumption_Zone2'] + df['PowerConsumption_Zone3']
        except Exception as e:
            print(f"Error loading data: {e}")
            df = pd.DataFrame()
    return df

def load_model():
    """Load the model if not already loaded"""
    global lstm_model, model_loaded
    if not model_loaded:
        try:
            lstm_model = PowerConsumptionLSTM()
            if lstm_model.load_model('models/power_consumption_lstm.h5'):
                model_loaded = True
                print("Successfully loaded pre-trained model")
            else:
                print("Error: Could not load pre-trained model")
                lstm_model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            lstm_model = None
    return lstm_model

def generate_graphs():
    """Generate visualization plots for the dashboard"""
    try:
        if df is None or df.empty:
            return None

        # 1. Time Series Plot
        time_series = go.Figure()
        time_series.add_trace(go.Scatter(
            x=df['Datetime'].tolist(),
            y=df['total_power'].tolist(),
            mode='lines',
            name='Historical Consumption'
        ))
        time_series.update_layout(
            title='Historical Power Consumption',
            xaxis_title='Time',
            yaxis_title='Power Consumption (kW)',
            height=400
        )

        # 2. Temperature vs Consumption
        temp_vs_consumption = go.Figure()
        temp_vs_consumption.add_trace(go.Scatter(
            x=df['Temperature'].tolist(),
            y=df['total_power'].tolist(),
            mode='markers',
            name='Temperature vs Consumption'
        ))
        temp_vs_consumption.update_layout(
            title='Temperature vs Power Consumption',
            xaxis_title='Temperature (°C)',
            yaxis_title='Power Consumption (kW)',
            height=300
        )

        # 3. Hourly Pattern
        df['hour'] = pd.to_datetime(df['Datetime']).dt.hour
        hourly_pattern = go.Figure()
        hourly_pattern.add_trace(go.Box(
            x=df['hour'].tolist(),
            y=df['total_power'].tolist(),
            name='Hourly Distribution'
        ))
        hourly_pattern.update_layout(
            title='Hourly Consumption Distribution',
            xaxis_title='Hour of Day',
            yaxis_title='Power Consumption (kW)',
            height=300
        )

        # Convert figures to JSON-serializable format
        return {
            'time_series': json.loads(time_series.to_json()),
            'temp_vs_consumption': json.loads(temp_vs_consumption.to_json()),
            'hourly_pattern': json.loads(hourly_pattern.to_json())
        }
    except Exception as e:
        print(f"Error generating graphs: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/estimate', methods=['POST'])
def estimate():
    try:
        # Load data and model
        global df, lstm_model
        df = load_data()
        lstm_model = load_model()
        
        if lstm_model is None:
            return jsonify({'error': 'Model not available'}), 500

        data = request.get_json()
        time_str = data.get('time')
        temperature = data.get('temperature')

        if not time_str:
            return jsonify({'error': 'Time is required'}), 400

        # Parse the time string
        try:
            time = datetime.strptime(time_str, '%H:%M:%S').time()
        except ValueError:
            return jsonify({'error': 'Invalid time format. Expected HH:MM:SS'}), 400

        if temperature is None:
            return jsonify({'error': 'Temperature is required'}), 400

        # Get the prediction using the loaded model
        prediction = lstm_model.predict(time, temperature)
        
        # Calculate cost (assuming $0.12 per kWh)
        cost = prediction * 0.12

        # Generate graphs
        graphs = generate_graphs()

        return jsonify({
            'estimated_consumption': float(prediction),
            'estimated_cost': float(cost),
            'graphs': graphs
        })

    except Exception as e:
        print(f"Error in estimate route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 10000
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 