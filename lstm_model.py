import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import gc

class PowerConsumptionLSTM:
    def __init__(self, sequence_length=12):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_names = ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows', 
                            'PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']
        self.is_fitted = False
        self.model_path = 'power_consumption_lstm.h5'
        self.scaler_path = 'scaler_params.npz'
        
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
        model.compile(optimizer='adam', loss='mse')
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
            # Save scaler parameters
            np.savez(self.scaler_path, 
                    scale_=self.scaler.scale_,
                    min_=self.scaler.min_,
                    data_min_=self.scaler.data_min_,
                    data_max_=self.scaler.data_max_,
                    data_range_=self.scaler.data_range_)
        
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
            
            # Save the model
            self.model.save(self.model_path)
            
            return history
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                # Load scaler parameters
                if os.path.exists(self.scaler_path):
                    scaler_params = np.load(self.scaler_path)
                    self.scaler.scale_ = scaler_params['scale_']
                    self.scaler.min_ = scaler_params['min_']
                    self.scaler.data_min_ = scaler_params['data_min_']
                    self.scaler.data_max_ = scaler_params['data_max_']
                    self.scaler.data_range_ = scaler_params['data_range_']
                    self.is_fitted = True
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, time_of_day, temperature):
        try:
            if not self.is_fitted:
                raise ValueError("Scaler has not been fitted. Please train the model first.")
            
            # Create a sample input with default values
            sample = np.zeros((1, len(self.feature_names)))  # One sample with all features
            
            # Fill in known values
            sample[0, 0] = temperature  # Temperature is the first feature
            # Set other features to mean values (this is a simplification)
            sample[0, 1] = 75.0  # Average humidity
            sample[0, 2] = 5.0   # Average wind speed
            sample[0, 3] = 100.0 # Average general diffuse flows
            sample[0, 4] = 100.0 # Average diffuse flows
            
            # Scale the input
            scaled_sample = self.scaler.transform(sample)
            
            # Create sequence by repeating the scaled sample
            sequence = np.tile(scaled_sample[:, :-3], (1, self.sequence_length, 1))
            
            # Make prediction
            prediction = self.model.predict(sequence, verbose=0)
            
            # Inverse transform the prediction
            dummy = np.zeros((1, len(self.feature_names)))
            dummy[0, -3:] = prediction[0]  # Put prediction in the last 3 columns
            prediction = self.scaler.inverse_transform(dummy)[0, -3:]
            
            return prediction
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
    
    def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
        gc.collect() 