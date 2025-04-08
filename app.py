from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lstm_model import PowerConsumptionLSTM
import os
import plotly.graph_objects as go
import plotly.express as px
import json

app = Flask(__name__)

# Load the power consumption data and initialize model
try:
    df = pd.read_csv('powerconsumption.csv')
    # Calculate total power consumption
    df['total_power'] = df['PowerConsumption_Zone1'] + df['PowerConsumption_Zone2'] + df['PowerConsumption_Zone3']
    
    # Initialize LSTM model
    lstm_model = PowerConsumptionLSTM()
    
    # Train the model if it doesn't exist or if scalers aren't fitted
    if not lstm_model.load_model() or not lstm_model.is_fitted:
        print("Training LSTM model (optimized for speed)...")
        lstm_model.train(df, epochs=20)  # Reduced epochs
    else:
        print("Loaded existing LSTM model")
        
except Exception as e:
    print(f"Error loading data or training model: {e}")
    df = pd.DataFrame()
    lstm_model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/estimate', methods=['POST'])
def estimate():
    try:
        data = request.get_json()
        
        # Extract input parameters
        time_of_day = data.get('time_of_day')
        temperature = float(data.get('temperature', 20))  # Default to 20°C if not provided
        
        # Convert time string to datetime
        current_time = datetime.strptime(time_of_day, '%H:%M')
        
        # If we have a trained LSTM model, use it for prediction
        if lstm_model is not None and lstm_model.is_fitted:
            # Make prediction using LSTM model
            predictions = lstm_model.predict(current_time, temperature)
            
            # Sum up the predictions for all zones
            estimated_consumption = float(sum(predictions) / 1000)  # Convert to kW and to float
            
            # Create graphs
            # 1. Time Series Plot
            time_series = go.Figure()
            time_series.add_trace(go.Scatter(
                x=df['Datetime'].tolist(),  # Convert to list
                y=df['total_power'].tolist(),  # Convert to list
                mode='lines',
                name='Historical Consumption'
            ))
            time_series.update_layout(
                title='Historical Power Consumption',
                xaxis_title='Time',
                yaxis_title='Power Consumption (kW)'
            )
            
            # 2. Temperature vs Consumption
            temp_vs_consumption = go.Figure()
            temp_vs_consumption.add_trace(go.Scatter(
                x=df['Temperature'].tolist(),  # Convert to list
                y=df['total_power'].tolist(),  # Convert to list
                mode='markers',
                name='Temperature vs Consumption'
            ))
            temp_vs_consumption.update_layout(
                title='Temperature vs Power Consumption',
                xaxis_title='Temperature (°C)',
                yaxis_title='Power Consumption (kW)'
            )
            
            # 3. Hourly Pattern
            df['hour'] = pd.to_datetime(df['Datetime']).dt.hour
            hourly_pattern = go.Figure()
            hourly_pattern.add_trace(go.Box(
                x=df['hour'].tolist(),  # Convert to list
                y=df['total_power'].tolist(),  # Convert to list
                name='Hourly Distribution'
            ))
            hourly_pattern.update_layout(
                title='Hourly Consumption Distribution',
                xaxis_title='Hour of Day',
                yaxis_title='Power Consumption (kW)'
            )
            
            # Convert figures to JSON-serializable format
            graphs = {
                'time_series': json.loads(time_series.to_json()),
                'temp_vs_consumption': json.loads(temp_vs_consumption.to_json()),
                'hourly_pattern': json.loads(hourly_pattern.to_json())
            }
            
        else:
            # Fallback to basic estimation if model is not available
            if 8 <= current_time.hour <= 18:
                base_consumption = 1200  # kW
            else:
                base_consumption = 800  # kW
            
            temp_factor = 1 + (temperature - 20) * 0.02
            estimated_consumption = base_consumption * temp_factor
            graphs = None
        
        # Calculate cost (assuming $0.12 per kWh)
        rate_per_kwh = 0.12
        estimated_cost = float(estimated_consumption * rate_per_kwh)  # Convert to float
        
        return jsonify({
            'success': True,
            'time_of_day': time_of_day,
            'temperature': float(temperature),  # Convert to float
            'estimated_consumption': round(float(estimated_consumption), 2),  # Convert to float
            'estimated_cost': round(float(estimated_cost), 2),  # Convert to float
            'units': 'kW',
            'graphs': graphs
        })
        
    except Exception as e:
        print(f"Error in estimate route: {str(e)}")  # Add debug print
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Changed port to 5001 