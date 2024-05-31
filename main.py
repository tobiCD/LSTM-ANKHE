import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint


# Read data from CSV
df = pd.read_csv('csvfile/DuLieuAnKhe2.csv')
df = pd.read_csv('csvfile/DuLieuAnKhe2.csv')
df.index = df['Time']
data = df.filter(['FlowWater'])
dataset = data.values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Split the data into training and testing sets
training_data_len = int(np.ceil(len(dataset) * 0.80))
train_data = scaled_data[0:int(training_data_len), :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
model.fit(x_train, y_train, batch_size=50, epochs=5)

# Test data preparation
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - y_test)**2))
print("Root Mean Squared Error (RMSE):", rmse)

# Plot the results
train = data[:training_data_len]
valid = data[training_data_len:].copy()
valid['Predictions'] = predictions


plt.figure(figsize=(16, 6))
plt.title('LSTM Model Prediction for FlowWater')
plt.xlabel('Time')
plt.ylabel('FlowWater')
plt.plot(train['FlowWater'], label='Training Data')
plt.plot(valid['FlowWater'], label='Actual Data', linewidth=2 )
plt.plot(valid['Predictions'], label='Predictions', linewidth=2)
plt.legend()
plt.tight_layout()
plt.show()
