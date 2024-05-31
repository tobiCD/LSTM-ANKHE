import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model

from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.python import keras

data = pd.read_csv('csvfile/DuLieuAnKhe2.csv')
data_end = int(np.floor(0.8 * (data.shape[0])))
train = data[0:data_end]['FlowWater']
data['Time'] = pd.to_datetime(data['Time'])
data.index = data['Time']
test = data[data_end:]['FlowWater'].values.reshape(-1)
date_test = data[data_end:]['Time'].values.reshape(-1)


def get_data(train, test, time_step, num_predict, date):
    x_train = list()
    y_train = list()
    x_test = list() 
    y_test = list()
    date_test = list()

    for i in range(0, len(train) - time_step - num_predict):
        x_train.append(train[i:i + time_step])
        y_train.append(train[i + time_step:i + time_step + num_predict])

    for i in range(0, len(test) - time_step - num_predict):
        x_test.append(test[i:i + time_step])
        y_test.append(test[i + time_step:i + time_step + num_predict])
        date_test.append(date[i + time_step:i + time_step + num_predict])

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), np.asarray(date_test)


""" time_step: Trong bài toán này thì bạn hiểu là nếu bạn muốn cứ 30 giá trị của Open thì đoán 1 giá trị open tiếp theo thì time_step ở đây bằng 30. Còn num_predict là 1.
 Từ đó ta thấy hàm get_data ở trên mục đích là định dạng lại dữ liệu để tí có thể đưa vào mạng. 
 Ví dụ, sau khi qua hàm get_data thì: x_train = [[1,2,3,4,5,6,7,8,9,10],[2,3,4,5,6,7,8,9,10,11]] và y_train = [11,12] 
 Bản chất là mình muốn dùng [1,2,3,4,5,6,7,8,9,10] để đoán ra 11, [2,3,4,5,6,7,8,9,10,11] để đoán ra 12"""

x_train, y_train, x_test, y_test, date_test = get_data(train, test, 30, 1, date_test)
# dua ve 0->1 cho tap train
scaler = MinMaxScaler()
x_train = x_train.reshape(-1, 2)
x_train = scaler.fit_transform(x_train)
y_train = scaler.fit_transform(y_train)

# đưa về 0-> 1 cho tập test
x_test = x_test.reshape(-1, 2)
x_test = scaler.fit_transform(x_test)
y_test = scaler.fit_transform(y_test)

"""Ở đây, mình đã code theo kiểu lấy 30 dữ liệu để đoán 1 dữ liệu tiếp theo.
 Sau đó mình chuẩn hóa dữ liệu về dạng từ 0 đến 1, theo hàm MinMaxScaler() cho bộ train và test. 
 Mục đích của chuẩn hóa là để tí nữa vào mô hình mạng nó tối ưu tốt hơn.
  Tiếp theo, chúng ta sẽ reshape lại cho x_train và y_train :
"""

# reshape cho đúng model
x_train = x_train.reshape(-1, 30, 1)
y_train = y_train.reshape(-1, 1)

# reshape lai cho test
x_test = x_test.reshape(-1, 30, 1)
y_test = y_test.reshape(-1, 1)
date_test = date_test.reshape(-1, 1)
"""Batch size: Cứ hiểu là có bao nhiêu cặp (time_steps, feature) ấy time_steps:
 Như trình bày ở trên rồi feature: là có bao nhiêu thuộc tính của mỗi phần tử trong time_step.
  Ví dụ : time_step có 10 giá trị (mỗi giá trị là một vector),
  mỗi vector là một giá trị 2 chiều chẳng hạn, thì feature ở đây là 2 (tức 2 chiều đó ).
   Tóm cái váy lại, thì feature cứ hiểu là số thuộc tính của mỗi phần tử time_step. 
   Còn reshape đầu ra mục đích là tí cho hợp với shape đầu ra của mô hình mạng. 
Ở trên ta thấy dùng 30 để đoán 1, nên đầu ra ở đây phải reshape theo (-1,1)."""

# đầu vòa 30 đoán 1
n_input = 30
n_feature = 1

model = Sequential()
model.add(LSTM(units=50,activation='relu', input_shape = (n_input , n_feature) , return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=50 , return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=50))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(optimizer='adam',loss = 'mse')

model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=1, batch_size=50)


test_output = model.predict(x_test)

# print(test_output)
test_1 = scaler.inverse_transform(test_output)
test_2=scaler.inverse_transform(y_test)
rmse = np.sqrt(mean_squared_error(test_2, test_1))
r2 = r2_score(test_2, test_1)
mse = mean_squared_error(test_2, test_1)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")
print(f"Mean Squared Error (MSE): {mse}")
# plt.plot(test_1[:test_output], color='r')
# plt.plot(test_2[:y_test] ,color='b')
plt.figure(figsize=(16, 6))

plt.plot(test_1, color='r', label='Prediction')
plt.plot( test_2, color='b', label='Reality')
plt.title("FlowWater")
plt.xlabel("Time")
plt.ylabel("Flowwater")
plt.legend(('prediction', 'reality'),loc='upper right')
plt.show()



