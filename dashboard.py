import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lstm_model import PowerConsumptionLSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

try:
    # Load and prepare data
    df = pd.read_csv('powerconsumption.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Calculate total power for visualization
    df['total_power'] = df['PowerConsumption_Zone1'] + df['PowerConsumption_Zone2'] + df['PowerConsumption_Zone3']
    
    # Initialize LSTM model
    lstm_model = PowerConsumptionLSTM()
    
    # Train model and get predictions for visualization
    if not lstm_model.load_model():
        print("Training LSTM model for dashboard (optimized for speed)...")
        lstm_model.train(df, epochs=20)  # Reduced epochs
        lstm_model.save_model()
    else:
        print("Loaded existing LSTM model")
    
    # Prepare data for model evaluation
    X, y = lstm_model.prepare_data(df)
    predictions = lstm_model.model.predict(X)
    predictions = lstm_model.scaler.inverse_transform(predictions)
    actual_values = lstm_model.scaler.inverse_transform(y)
    
    # Calculate metrics
    mse = mean_squared_error(actual_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, predictions)
    r2 = r2_score(actual_values, predictions)
    
    # Create layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Power Consumption Analysis Dashboard", className="text-center mb-4")
            ])
        ]),
        
        # Metrics Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Model Accuracy Metrics", className="card-title"),
                        html.P(f"RMSE: {rmse:.2f} kW"),
                        html.P(f"MAE: {mae:.2f} kW"),
                        html.P(f"R² Score: {r2:.2f}")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Data Overview", className="card-title"),
                        html.P(f"Total Records: {len(df):,}"),
                        html.P(f"Date Range: {df['Datetime'].min().date()} to {df['Datetime'].max().date()}"),
                        html.P(f"Average Consumption: {df['total_power'].mean():.2f} kW")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Peak Analysis", className="card-title"),
                        html.P(f"Peak Consumption: {df['total_power'].max():.2f} kW"),
                        html.P(f"Peak Time: {df.loc[df['total_power'].idxmax(), 'Datetime'].strftime('%Y-%m-%d %H:%M')}"),
                        html.P(f"Average Peak Hours (8-18): {df[df['Datetime'].dt.hour.between(8, 18)]['total_power'].mean():.2f} kW")
                    ])
                ])
            ], width=4)
        ], className="mb-4"),
        
        # Time Series Plot
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Power Consumption Over Time", className="card-title"),
                        dcc.Graph(
                            figure=px.line(df, x='Datetime', y='total_power',
                                         title='Historical Power Consumption')
                        )
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Model Performance
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Model Predictions vs Actual Values", className="card-title"),
                        dcc.Graph(
                            figure=go.Figure()
                            .add_trace(go.Scatter(y=actual_values.flatten(), name='Actual'))
                            .add_trace(go.Scatter(y=predictions.flatten(), name='Predicted'))
                            .update_layout(title='Model Performance')
                        )
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Prediction Error Distribution", className="card-title"),
                        dcc.Graph(
                            figure=px.histogram(
                                x=actual_values.flatten() - predictions.flatten(),
                                title='Prediction Error Distribution'
                            )
                        )
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Temperature vs Consumption
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Temperature vs Power Consumption", className="card-title"),
                        dcc.Graph(
                            figure=px.scatter(df, x='Temperature', y='total_power',
                                            title='Temperature vs Power Consumption')
                        )
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Hourly Consumption Pattern", className="card-title"),
                        dcc.Graph(
                            figure=px.box(df, x='hour', y='total_power',
                                        title='Hourly Consumption Distribution')
                        )
                    ])
                ])
            ], width=6)
        ], className="mb-4")
    ], fluid=True)

except Exception as e:
    print(f"Error initializing dashboard: {e}")
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Error Loading Dashboard", className="text-center mb-4"),
                html.Div([
                    html.P(f"An error occurred: {str(e)}"),
                    html.P("Please check the console for more details.")
                ], className="alert alert-danger")
            ])
        ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050) 