import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('csvfile/DuLieuAnKhe2.csv')
data['Time'] = pd.to_datetime(data['Time'])
data.index = data['Time']

# Train-test split
data_end = int(np.floor(0.8 * (data.shape[0])))
train = data[0:data_end]['FlowWater']
test = data[data_end:]['FlowWater'].values.reshape(-1)
date_test = data[data_end:]['Time'].values.reshape(-1)

# Function to get data for modeling
def get_data(train, test, time_step, num_predict, date):
    x_train, y_train, x_test, y_test, date_test = [], [], [], [], []

    for i in range(0, len(train) - time_step - num_predict):
        x_train.append(train[i:i + time_step])
        y_train.append(train[i + time_step:i + time_step + num_predict])

    for i in range(0, len(test) - time_step - num_predict):
        x_test.append(test[i:i + time_step])
        y_test.append(test[i + time_step:i + time_step + num_predict])
        date_test.append(date[i + time_step:i + time_step + num_predict])

    return (
        np.asarray(x_train),
        np.asarray(y_train),
        np.asarray(x_test),
        np.asarray(y_test),
        np.asarray(date_test),
    )

# Get data for modeling
x_train, y_train, x_test, y_test, date_test = get_data(train, test, 30, 1, date_test)

# Normalize the data using a single scaler for both training and testing
scaler = MinMaxScaler()
x_train = x_train.reshape(-1,30)

x_train = scaler.fit_transform(x_train)
y_train = scaler.fit_transform(y_train)

# dua ve 0->1 cho tap test
x_test = x_test.reshape(-1,30)

x_test = scaler.fit_transform(x_test)
y_test = scaler.fit_transform(y_test)

# Reshape for model input
x_train = x_train.reshape(-1, 30, 1)
y_train = y_train.reshape(-1, 1)

x_test = x_test.reshape(-1, 30, 1)
y_test = y_test.reshape(-1, 1)
date_test = date_test.reshape(-1, 1)

# Define model architecture
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(30, 1), return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=50))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=1, batch_size=30)

# Predictions
test_output = model.predict(x_test)
test_output = scaler.inverse_transform(test_output)
y_test_inv = scaler.inverse_transform(y_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test_inv, test_output))
r2 = r2_score(y_test_inv, test_output)
mse = mean_squared_error(y_test_inv, test_output)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")
print(f"Mean Squared Error (MSE): {mse}")

# Plotting
plt.plot(test_output[:1000], color='r', label='Prediction')
plt.plot(y_test_inv[:1000], color='b', label='Reality')
plt.title("FlowWater Prediction")
plt.xlabel("Time")
plt.ylabel("FlowWater")
plt.legend(loc='upper right')
plt.show()
