import pandas as pd
import numpy as np

def load_info(path):
    info = pd.read_csv(path)
    info.set_index('ID', inplace=True)

    # replace blanks
    for j in range(info.shape[1]):
        for i in range(info.shape[0]):
            try:
                int(info.iloc[i, j])
            except:
                info.iloc[i, j] = -1

    for j in range(info.shape[1]):
        info.iloc[:, j] = info.iloc[:, j].astype(int)

    info['count'] = np.zeros((info.shape[0]), dtype=int)
    info['children'] = np.zeros((info.shape[0]), dtype=int)

    # 세대원 수가 1명인 경우
    info.loc[info['Q410'] == 1, 'count'] = 1
    info.loc[info['Q410'] == 1, 'children'] = 0

    # 모든 세대원이 15살 이상인 경우
    info.loc[info['Q410'] == 2, 'count'] = info.loc[info['Q410'] == 2, 'Q420']
    info.loc[info['Q410'] == 2, 'children'] = 0

    # children이 있는 경우
    info.loc[info['Q410'] == 3, 'count'] = info.loc[info['Q410'] == 3, 'Q420'] + info.loc[info['Q410'] == 3, 'Q43111']
    info.loc[info['Q410'] == 3, 'children'] = info.loc[info['Q410'] == 3, 'Q43111']
    return info


## 데이터 로드 및 전처리
power_df = pd.read_csv('data/power_comb.csv')
# 0 to NaN
power_df[power_df==0] = np.nan
# power_df = power_df.iloc[:2*24*28,:] # 4주
info = load_info(path='data/survey_for_number_of_residents.csv')
datetime = power_df['time']
datetime = pd.to_datetime(datetime)
power_df.set_index('time', inplace=True)
# info[info>=7] = 7
label = info['count'].values
label = label.astype(int)

# 2009년 9월 1일 ~2010년 12월 31일
start = 48 * 49
end = start + 48 * 487
power_df = power_df.iloc[start:end,:]
datetime = datetime[start:end]


#%% 사람수별로 대표 데이터 만듦
rep_data = dict()
# label에 따라 대표 데이터 load
for v in np.unique(label):
    idx = v == label
    data = np.nanmean(power_df.iloc[:,idx], axis=1)
    rep_data[v] = data
del rep_data[9]

rep_data[9] = np.nanmean([rep_data[8], rep_data[10]], axis=0)
rep_data[11] = np.nanmean([rep_data[10], rep_data[12]], axis=0)

rep_data = pd.DataFrame.from_dict(rep_data)
rep_data = rep_data.interpolate('spline', order=1)


#%%
# 호텔 데이터 load
hotel_data = pd.read_csv('data/City Hotel.csv')
hotel_data = hotel_data.iloc[52:,:].reset_index(drop=True)
hotel_data = hotel_data.iloc[:487,:].reset_index(drop=True)
datetime_hotel = pd.to_datetime(hotel_data['Unnamed: 0'])

#%% 호텔데이터에 해당하는 전력데이터 augmentation
power_data_hotel = pd.DataFrame(index = datetime_hotel, columns=range(1,49))

for d in range(hotel_data.shape[0]):
    value = np.zeros(487 * 48)
    for i in range(1,13):
        value += hotel_data.loc[d,str(i)] * rep_data[i].values
    value = value.reshape(-1,48)
    power_data_hotel.iloc[d,:] = value[d,:]

# 여러 피쳐 추가
power_data_hotel['mean'] = power_data_hotel.mean(axis=1)

#%%
import matplotlib.pyplot as plt
label = hotel_data['sum']
power_data_hotel['sum'] = label.values

# correlation
print(np.corrcoef(power_data_hotel['mean'], power_data_hotel['sum']))

plt.plot(power_data_hotel['mean'], power_data_hotel['sum'],'.',markersize=12, color=[0.5,0.5,0.5])
plt.xlabel('하루 평균 전력 사용량 [kWh]')
plt.ylabel('투숙객 수')

plt.show()

#%% 학습데이터 준비
def trans(dataset, pasts, future, x_col_index, y_col_index):
    pasts_rev = np.insert(pasts+1, 0, 0)
    data_agg = np.zeros((dataset.shape[0]-pasts.max()-future,pasts.sum()+len(pasts)))
    labels = np.zeros((dataset.shape[0]-pasts.max()-future, future))
    strat, end = 0, dataset.shape[0]
    for j, x_col in enumerate(x_col_index):
        strat = strat + pasts[j]
        data = []
        dataset_sub = dataset[:, x_col]
        for i in range(strat, end - future):
            indices = np.array(dataset_sub[i - pasts[j]:i+1])
            data.append(indices)
        data = np.array(data)
        data = data.reshape(data.shape[0], -1)
        data = data[max(pasts) - pasts[j]:, :]
        data_agg[:,pasts_rev[:j+1].sum():pasts_rev[:j+2].sum()] = data
        strat = 0
    for j, i in enumerate(range(max(pasts), end - future)):
        labels[j,:] = np.array(dataset[i+1:i + future+1, y_col_index])
    # labels = np.ravel(labels)
    return data_agg, labels

x_columns = ['mean',  'sum']
y_column = 'sum'

future = 7
pasts = np.zeros(len(x_columns)).astype(int)
pasts[0] = 7

x_col_index = np.zeros(np.shape(x_columns),dtype=int)
for i, x_column in enumerate(x_columns):
    x_col_index[i] = np.where(x_column == power_data_hotel.columns)[0][0]
y_col_index = np.where(y_column == power_data_hotel.columns)[0][0]

data, label = trans(power_data_hotel.values, pasts, future, x_col_index, y_col_index)

TRAIN_SPLIT = 365
X_train, X_test = data[:TRAIN_SPLIT,:], data[TRAIN_SPLIT:]
y_train, y_test = label[:TRAIN_SPLIT], label[TRAIN_SPLIT:]
y_train = np.ravel(y_train[:,-1])
y_test = np.ravel(y_test[:,-1])

# 학습
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smape(true, pred):
    true = np.ravel(true)
    pred = np.ravel(pred)
    v = 2 * abs(pred - true) / (abs(pred) + abs(true))
    output = np.mean(v) * 100
    return output

from sklearn.svm import SVR

############## w/o electricity
model = SVR(kernel='linear')
model.fit(X_train[:,-1].reshape(-1, 1), y_train)

# 예측 - self로
y_pred_self = model.predict(X_test[:,-1].reshape(-1, 1))
print('w/o electricity: {:.2f}'.format(smape(y_test, y_pred_self)))

############## w/ electricity
model = SVR(kernel='linear')
model.fit(X_train, y_train)

# 예측 - 전력만 가지고
y_pred_multi = model.predict(X_test)
print('w/ electricity: {:.2f}'.format(smape(y_test, y_pred_multi)))


#%% 전력만으로 예측
data = power_data_hotel['mean'].values.reshape(-1,1)
label = power_data_hotel['sum'].values.reshape(-1,1)
X_train, X_test = data[:TRAIN_SPLIT,:], data[TRAIN_SPLIT:]
y_train, y_test = label[:TRAIN_SPLIT], label[TRAIN_SPLIT:]

model = SVR(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('only w/ electricity: {:.2f}'.format(smape(y_test,y_pred)))

#%% 앙상블
y_pred_ens = y_pred[14:] * 0.2 + y_pred_self * 0.8

print('w/o electricity: {:.2f}'.format(smape(y_test, y_pred_self)))
print('only electricity: {:.2f}'.format(smape(y_test, y_pred[14:])))
print('w/ electricity: {:.2f}'.format(smape(y_test, y_pred_ens)))


#%% 시각화
import matplotlib
font = {'size': 15, 'family':"NanumGothic"}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
# 학습 데이터
# plt.plot(datetime_hotel[300:],power_data_hotel['sum'][300:], label='실제값')
# plt.plot(date[TRAIN_SPLIT+1:],y_test, label='실제값')
# plt.plot(datetime_hotel[TRAIN_SPLIT+1:], y_pred_self,'--', label='전력데이터 미포함')
# plt.plot(datetime_hotel[TRAIN_SPLIT+1:], y_pred_multi,'--', label='전력데이터 포함')
# shape_ = y_pred_self.shape[0]

# plt.plot(datetime_hotel[-74:-70],y_test[-74:-70], label='실제값')
# plt.plot(datetime_hotel[-74:-70],y_pred_self[-74:-70],'-.', label='모델 1')
# plt.plot(datetime_hotel[-74:-70],y_pred[-74:-70],'-.', label='모델 2')
# plt.plot(datetime_hotel[-74:-70],y_pred_ens[-74:-70],'--', label='모델 3')
# plt.xticks(datetime_hotel[-74:-70],datetime_hotel[-74:-70].dt.strftime("%Y-%m-%d"))
day1 = 73
day2 = 65
plt.figure(figsize=(6,4))
plt.plot(datetime_hotel[-day1:-day2],y_test[-day1:-day2], label='실제값')
plt.plot(datetime_hotel[-day1:-day2],y_pred_self[-day1:-day2],'-.', label='모델 1', color='tab:green')
plt.plot(datetime_hotel[-day1:-day2],y_pred_ens[-day1:-day2],'r--', label='모델 2')
# plt.xticks(datetime_hotel[-day1:-day2],datetime_hotel[-day1:-day2].dt.strftime("%Y-%m-%d"))
plt.xticks(rotation=30)
plt.xlabel('날짜')
plt.ylabel('투숙객 수')
plt.legend(loc='upper left')
plt.show()


#%% 상관관계 그래프
