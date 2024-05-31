import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load data from CSV
df = pd.read_csv('csvfile/DuLieuAnKhe2 ver2  (1).csv', parse_dates=['Thời gian'], index_col='Thời gian')

# Use multiple columns for prediction
data = df[['Lưu lượng đến hồ (m³/s)']]

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Scale the data
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# Generate time series sequences
n_input = 12  # You can adjust this based on your preference
n_features = data.shape[1]
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# Define the LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(generator, epochs=10)

# Evaluate the model on the test set
test_generator = TimeseriesGenerator(scaled_test, scaled_test, length=n_input, batch_size=1)
predictions = []

for i in range(len(test_generator)):
    x, _ = test_generator[i]
    pred = model.predict(x)[0][0]
    predictions.append(pred)

# Invert scaling to get predictions in the original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Invert scaling for the actual test set
true_values = scaler.inverse_transform(scaled_test)

# Calculate RMSE
rmse = sqrt(mean_squared_error(true_values[n_input:], predictions))
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size + n_input:], true_values[n_input:], label='Actual')
plt.plot(df.index[train_size + n_input:], predictions, label='Predicted')
plt.title('Lưu lượng nước đến hồ - Actual vs Predicted')
plt.xlabel('Thời gian')
plt.ylabel('Lưu lượng đến hồ (m³/s)')
plt.legend()
plt.show()

# Generate future time series sequences for the next month
future_sequence = scaled_test[-n_input:]  # Take the last sequence from the test set
future_predictions = []

# Define the number of steps to predict into the future (e.g., 30 for 1 month)
future_steps = 30

for _ in range(future_steps):
    future_pred = model.predict(np.expand_dims(future_sequence, axis=0))[0][0]
    future_predictions.append(future_pred)
    future_sequence = np.concatenate((future_sequence[1:], [[future_pred]]))  # Shift the sequence

# Invert scaling for future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size + n_input:], true_values[n_input:], label='Actual')
plt.plot(pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D'), future_predictions, label='Future Predictions')
plt.title('Lưu lượng nước đến hồ - Actual vs Predicted vs Future Predictions')
plt.xlabel('Thời gian')
plt.ylabel('Lưu lượng đến hồ (m³/s)')
plt.legend()
plt.show()
