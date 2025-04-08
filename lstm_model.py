import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import gc
import joblib

class PowerConsumptionLSTM:
    def __init__(self, sequence_length=12):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_names = ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows', 
                            'PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']
        self.is_fitted = False
        
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length), :-3])  # All features except power consumption
            y.append(data[i + self.sequence_length, -3:])  # Power consumption for all three zones
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(3)  # Output layer for 3 zones
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')  # Using full name instead of 'mse'
        return model
    
    def prepare_data(self, df):
        # Ensure all required columns are present
        if not all(col in df.columns for col in self.feature_names):
            raise ValueError("Missing required columns in the dataset")
        
        # Select features in the correct order
        features = df[self.feature_names].values
        
        # Fit scaler if not already fitted
        if not self.is_fitted:
            self.scaler.fit(features)
            self.is_fitted = True
        
        # Transform features
        scaled_features = self.scaler.transform(features)
        return self.create_sequences(scaled_features)
    
    def train(self, df, epochs=20, batch_size=64, validation_split=0.2):
        try:
            X, y = self.prepare_data(df)
            
            # Build and train model
            self.model = self.build_model((X.shape[1], X.shape[2]))
            
            # Add early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=1
            )
            
            return history
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def save_model(self, model_path):
        """Save both the Keras model and the scaler"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save Keras model
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.save')
        joblib.dump(self.scaler, scaler_path)
    
    def load_model(self, model_path):
        """Load both the Keras model and the scaler"""
        try:
            # Load Keras model
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                
                # Load scaler
                scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.save')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    self.is_fitted = True
                    return True
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, time, temperature):
        """
        Make a prediction for a given time and temperature
        
        Args:
            time (datetime.time): Time of day
            temperature (float): Temperature in Celsius
            
        Returns:
            float: Predicted power consumption in kW
        """
        try:
            if not self.is_fitted or self.model is None:
                print("Model not properly loaded")
                return 0.0

            # Create a sample input with default values
            sample = np.zeros((1, len(self.feature_names)))
            
            # Set known values
            sample[0, 0] = temperature  # Temperature
            
            # Set other features to reasonable default values
            sample[0, 1] = 75.0  # Humidity (average value)
            sample[0, 2] = 5.0   # WindSpeed (average value)
            sample[0, 3] = 100.0 # GeneralDiffuseFlows (average value)
            sample[0, 4] = 100.0 # DiffuseFlows (average value)
            
            # Scale the features
            scaled_features = self.scaler.transform(sample)
            
            # Create sequence by repeating the scaled features
            sequence = np.tile(scaled_features[:, :-3], (1, self.sequence_length, 1))
            
            # Make prediction
            predictions = self.model.predict(sequence, verbose=0)
            
            # Create dummy array for inverse transform
            dummy = np.zeros((1, len(self.feature_names)))
            dummy[0, -3:] = predictions[0]  # Last 3 columns are power consumption predictions
            
            # Inverse transform to get actual values
            result = self.scaler.inverse_transform(dummy)[0, -3:]  # Get power consumption values
            
            # Sum up the predictions for all zones and convert to kW
            total_power = float(np.sum(result)) / 1000.0  # Convert to kW
            
            return total_power
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return 0.0  # Return 0 as fallback
    
    def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
        gc.collect() 