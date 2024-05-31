import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from math import sqrt

# Đọc dữ liệu từ CSV
df = pd.read_csv("DuLieuAnKhe2.csv")

# Chuẩn bị thời gian
df['Time'] = pd.to_datetime(df['Time'])
df['Time'] = (df['Time'] - df['Time'].min()).dt.total_seconds() / 3600

# Chuẩn hóa dữ liệu đầu vào và đầu ra
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

df[['Time', 'WaterLevel', 'Total']] = scaler_X.fit_transform(df[['Time', 'WaterLevel', 'Total']])
df['FlowWater'] = scaler_y.fit_transform(df['FlowWater'].values.reshape(-1, 1))
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data)-seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)
# Chọn độ dài của chuỗi thời gian
seq_length = 10


# Tạo chuỗi thời gian cho dữ liệu đầu vào và đầu ra
X = create_sequences(df[['Time', 'WaterLevel', 'Total']].values, seq_length)
y = create_sequences(df['FlowWater'].values, seq_length)

# Chuẩn bị dữ liệu đầu vào và đầu ra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình LSTM
model = Sequential()
model.add(LSTM(128, activation="tanh", input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation="tanh", return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, activation="tanh", return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(16, activation="tanh"))
model.add(Dropout(0.2))
model.add(Dense(1))

# Cấu hình mô hình
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mse")

# Hàm lịch trình để điều chỉnh tỷ lệ học
def scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        return lr * 0.9
    else:
        return lr * 1.0

lr_schedule = LearningRateScheduler(scheduler)

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Huấn luyện mô hình với Early Stopping
model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=[lr_schedule, early_stopping], validation_data=(X_test, y_test))

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Chuyển đổi dự đoán về dạng ban đầu (đã chuẩn hóa)
y_pred_reshaped = scaler_y.inverse_transform(y_pred[:, -1].reshape(-1, 1))

# Tính RMSE và NSE
rmse = np.sqrt(mean_squared_error(y_test[:, -1], y_pred_reshaped))
nse = 1 - np.sum((y_pred_reshaped - y_test[:, -1])**2) / np.sum((y_test[:, -1] - np.mean(y_test[:, -1]))**2)

# In thông số đánh giá
print("RMSE:", rmse)
print("NSE:", nse * 100, "%")

# Vẽ biểu đồ kết quả dự đoán
plt.figure(figsize=(12, 6))
plt.plot(df['Time'].iloc[-len(y_test):], scaler_y.inverse_transform(y_test[:, -1].reshape(-1, 1)), label='Thực tế')
plt.plot(df['Time'].iloc[-len(y_test):], y_pred_reshaped, label='Dự đoán')
plt.xlabel('Thời gian')
plt.ylabel('FlowWater')
plt.legend()
plt.show()
