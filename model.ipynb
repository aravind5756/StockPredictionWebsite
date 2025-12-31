import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Download stock data
ticker = "MSFT"
end_date = datetime.now()
start_date = end_date - timedelta(days=11*365)

stock_data = yf.download(ticker, start=start_date, end=end_date)

# Use Close price
data = stock_data['Close'].values.reshape(-1, 1)
dates = stock_data.index

# Split: First 10 years training, last year testing
split_idx = len(data) - 252  # Last 252 days for testing
train_data = data[:split_idx]
test_data = data[split_idx:]
train_dates = dates[:split_idx]
test_dates = dates[split_idx:]

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Create sequences for LSTM
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train).unsqueeze(-1)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test).unsqueeze(-1)
y_test = torch.FloatTensor(y_test)

# Define LSTM Model
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)
        # Take last output
        last_output = lstm_out[:, -1, :]
        # Final prediction
        prediction = self.fc(last_output)
        return prediction.squeeze()

# Initialize model
model = LSTMPredictor(input_size=1, hidden_size=50, num_layers=2, dropout=0.2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 100
batch_size = 32

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        avg_loss = epoch_loss / (len(X_train) / batch_size)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

# Predictions
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test).cpu().numpy()

# Convert back to original scale
y_test = y_test.numpy()
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_original = scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()

# Adjust dates (remove first seq_length days from test period)
test_dates_adjusted = test_dates[seq_length:]

# Calculate metrics
mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
r2 = r2_score(y_test_original, y_pred_original)
mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100

print(f"\n=== Model Performance on Last Year ===")
print(f"MAE: ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Directional accuracy
actual_direction = np.sign(y_test_original[1:] - y_test_original[:-1])
pred_direction = np.sign(y_pred_original[1:] - y_pred_original[:-1])
directional_accuracy = np.mean(actual_direction == pred_direction) * 100

price_accuracy = 100 - mape

print(f"\n=== ACCURACY SUMMARY ===")
print(f"Price Prediction Accuracy: {price_accuracy:.2f}%")
print(f"\n✓ Directional Accuracy: {directional_accuracy:.2f}%")
print(f"\n✓ R² Score: {r2:.2f}")

print(f"\n=== VOLATILITY ANALYSIS ===")
actual_volatility = np.std(y_test_original)
pred_volatility = np.std(y_pred_original)
print(f"Actual price volatility: ${actual_volatility:.2f}")
print(f"Predicted price volatility: ${pred_volatility:.2f}")
print(f"Volatility capture: {(pred_volatility/actual_volatility)*100:.1f}%")

# Weekly accuracy (better metric for LSTM)
weekly_actual = []
weekly_pred = []
for i in range(0, len(y_test_original), 5):
    weekly_actual.append(np.mean(y_test_original[i:i+5]))
    weekly_pred.append(np.mean(y_pred_original[i:i+5]))

weekly_mape = np.mean(np.abs((np.array(weekly_actual) - np.array(weekly_pred)) / np.array(weekly_actual))) * 100
print(f"\n=== WEEKLY PREDICTION ACCURACY ===")
print(f"Weekly MAPE: {weekly_mape:.2f}%")
print(f"Weekly Accuracy: {100-weekly_mape:.2f}%")

# Plot
plt.figure(figsize=(15, 6))
plt.plot(test_dates_adjusted, y_test_original, label='Actual Price', linewidth=2, color='blue')
plt.plot(test_dates_adjusted, y_pred_original, label='Predicted Price', linewidth=2, color='red', alpha=0.7)
plt.title(f'{ticker} Stock Price: Actual vs Predicted (LSTM Model)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
