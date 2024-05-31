import pandas as pd

df = pd.read_csv('csvfile/AnKheData_GA_SVR2.csv')

# Tạo cột 'Time' và gán giá trị cho nó
df['Time'] = pd.date_range(start='2016-01-01', periods=len(df))
time_column = df['Time']
df.drop(labels=['Time'], axis=1, inplace=True)  # Loại bỏ cột 'Time' từ vị trí hiện tại
df.insert(0, 'Time',time_column )
# Lưu lại dữ liệu đã xử lý
df.to_csv('AnKheData_GA_SVR3.csv', index=False)