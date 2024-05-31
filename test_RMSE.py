import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import datetime as dt

# Đọc dữ liệu từ CSV
df = pd.read_csv("DuLieuAnKhe/3.csv")

df['Ngày'] = pd.to_datetime(df['LVS/Hồ chứa/Ngày'])
df['Giờ'] = pd.to_timedelta(df['LVS/Hồ/Ngày/Giờ'])
df['Thời Gian'] = df['Ngày'] + df['Giờ']
df['Thời Gian'] = pd.to_datetime(df['Thời Gian'])
df['Thời Gian'] = (df['Thời Gian'] - df['Thời Gian'].min()).dt.total_seconds() / 3600
# Chọn cột cần dự đoán (Mực nước hồ) làm biến phụ thuộc (y)
y = df['Mực nước hồ (m)'].values
y = df['Mực nước hồ (m)'].values


# Chọn các cột cần sử dụng làm biến độc lập (X)
X = df[['Thời Gian', 'Lưu lượng đến hồ (m³/s)', 'Tổng lưu lượng xả (m³/s)[Thực tế]']].values

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Reshape dữ liệu để phù hợp với đầu vào của LSTM
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Tạo mô hình LSTM
model = Sequential()
model.add(LSTM(128, activation="tanh", input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dense(1))

# Cấu hình mô hình
model.compile(optimizer="adam", loss="mse")

# Huấn luyện mô hình
model.fit(X_train_reshaped, y_train_scaled, epochs=100, verbose=1)

# Dự đoán trên tập kiểm tra
y_pred_scaled = model.predict(X_test_reshaped)

# Chuyển đổi dự đoán về dạng ban đầu (đã chuẩn hóa)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# In thông số đánh giá
loss = model.evaluate(X_test_reshaped, y_test_scaled, verbose=0)
print("Loss:", loss)
y_test_original = scaler_y.inverse_transform(y_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))

# Tính NSE
nse = 1 - np.sum((y_pred - y_test_original)**2) / np.sum((y_test_original - np.mean(y_test_original))**2)

# In kết quả
print("RMSE:", rmse, "m")
print("NSE:", nse * 100, "%")
#
# plt.plot(y_test, label='Thực tế', linewidth=2)
# plt.plot(y_pred, label='Dự đoán', linewidth=2,linestyle='dashed')  # Adjust linestyle as needed
#
# # Customizing the y-axis range
# plt.ylim(400, 470)
#
# plt.xlabel('Giờ')
# plt.ylabel('Mực nước hồ')
# plt.legend()
# plt.title('So sánh giữa giá trị thực tế và dự đoán')
# plt.grid(True)
# plt.show()