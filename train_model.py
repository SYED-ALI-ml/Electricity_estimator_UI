import pandas as pd
from lstm_model import PowerConsumptionLSTM
import os

print("Loading data...")
df = pd.read_csv('powerconsumption.csv')
df['total_power'] = df['PowerConsumption_Zone1'] + df['PowerConsumption_Zone2'] + df['PowerConsumption_Zone3']

print("Initializing model...")
lstm_model = PowerConsumptionLSTM()

print("Training model...")
history = lstm_model.train(df, epochs=20)

print("Training completed. Saving model...")
# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the model and scaler
lstm_model.save_model('models/power_consumption_lstm.h5')

print("Model saved successfully!")
print("You can now commit the 'models' directory to your repository.") 