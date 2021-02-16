import pandas as pd
import numpy as np

hotel_data = pd.read_csv('data/City Hotel.csv')
hotel_data = hotel_data.iloc[52:,:].reset_index(drop=True)

hotel_data['Unnamed: 0'] = pd.to_datetime(hotel_data['Unnamed: 0'])
TRAIN_SPLIT = int(round(hotel_data.shape[0] * 0.5))

train = hotel_data.iloc[:TRAIN_SPLIT,4].values
val = hotel_data.iloc[TRAIN_SPLIT:,4].values

date = pd.to_datetime(hotel_data['Unnamed: 0'])


#%% 학습
X_train = train[:-1].reshape(-1,1)
y_train = train[1:]

X_test = val[:-1].reshape(-1,1)
y_test = val[1:]

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

from sklearn.svm import SVR
import matplotlib.pyplot as plt

model = SVR(kernel='rbf',C=80,gamma=0.001)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
print('{:.2f}'.format(mean_absolute_percentage_error(y_test, y_pred)))

#%% 결과 plot
import matplotlib
font = {'size': 16, 'family':"NanumGothic"}
matplotlib.rc('font', **font)

# 학습 데이터
plt.plot(date,hotel_data.iloc[:,4].values, label='실제값')
# plt.plot(date[TRAIN_SPLIT+1:],y_test, label='실제값')
plt.plot(date[TRAIN_SPLIT+1:], y_pred,'--', label='예측값')
plt.xticks(rotation=30)
plt.xlabel('날짜')
plt.ylabel('이용객 수')
plt.legend()
plt.show()