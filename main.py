import pandas as pd
import numpy as np
from util import *
import datetime
from sklearn.model_selection import train_test_split
import lightgbm as lgb

seed = 777
window = 24

#%% load dataset
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
## 24시간을 맞춤
train = train.iloc[13:,:]
## index: datetime
train.index = pd.to_datetime(train['Time'])
test_time = pd.to_datetime(test['Time'])
test.index = pd.to_datetime(test['Time'])
train.drop(columns=['Time'], inplace=True)
test.drop(columns=['Time'], inplace=True)

def load_calendar(start_date,end_date):
    datas = []
    data = pd.read_csv('data/calendar2017.csv', header=None)
    datas.append(data)
    data = pd.read_csv('data/calendar2018.csv', header=None)
    datas.append(data)
    calendar = pd.concat(datas, axis=0, join='outer')
    calendar = calendar.reset_index(drop=True)
    calendar.columns = ['year','month','day','dayofweek','wnw']
    calendar['date'] = calendar['year'].astype(str)\
                       +'-'+calendar['month'].astype(str)\
                       +'-'+calendar['day'].astype(str)
    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar.index = calendar['date']
    calendar = calendar.loc[start_date:end_date, :]
    calendar = calendar.loc[:, 'wnw']
    return calendar

start_date, end_date = test.index[0],test.index[-1]
calendar = load_calendar(start_date, end_date)  # 1-nw, 0-w

#%% preprocessing
## NaN processing
train[train == 0] = np.nan
test[test == 0] = np.nan

# 일단 대충..
def interp(df, window = 2):
    for col in range(df.shape[1]):
        home = np.reshape(df.iloc[:,col].values,(-1,24))
        for row in range(home.shape[0]):
            nan_idx = np.isnan(home[row, :])
            # daytype = calendar[row].values
            if row < window:
                home[row, nan_idx] = np.nanmean(home[row:row + window + 1, nan_idx])
            elif row > home.shape[0]-window-1:
                home[row, nan_idx] = np.nanmean(home[row-window:row+1,nan_idx])
            else:
                home[row, nan_idx] = np.nanmean(home[row-window:row+window+1,nan_idx])
        df.iloc[:,col] = np.ravel(home)
    return df

test = interp(test)
# train = interp(train)

#%% 24시간 예측
def normalize(data):
    data = data / np.max(data)
    data = np.log(data)
    mean_ = np.mean(data)
    std_ = np.std(data)
    norm = (data - mean_) / std_
    return norm, mean_, std_


def trans(data, time, window=24):
    data_t = []
    label = np.zeros((data.shape[0] - window - window, window))
    for i in range(window, data.shape[0] - window):
        data_s = np.array(data[i - window : i])
        data_t.append(data_s)
    # save test data
    test_data = np.array(data[data.shape[0] - window: ]).reshape(1,-1)
    data_t = np.array(data_t)
    for j, i in enumerate(range(window, data.shape[0] - window)):
        label[j,:] = np.array(data[i:i + window])

    # add time feature
    time_features = []
    time_s = time.iloc[:-48]
    time_features.append(time_s.dt.dayofweek)
    time_features.append(time_s.dt.quarter)
    time_features.append(time_s.dt.month)
    time_features.append(time_s.dt.year)
    time_features.append(time_s.dt.dayofyear)
    time_features.append(time_s.dt.day)
    time_features.append(time_s.dt.weekofyear)
    time_features = np.array(time_features).T
    data_t = np.concatenate([data_t,time_features], axis=1)

    # add time for test
    time_features = []
    time_s = time.iloc[-24]
    time_features.append(time_s.dayofweek)
    time_features.append(time_s.quarter)
    time_features.append(time_s.month)
    time_features.append(time_s.year)
    time_features.append(time_s.dayofyear)
    time_features.append(time_s.day)
    time_features.append(time_s.weekofyear)
    time_features = np.array(time_features).T
    time_features = time_features.reshape(1,-1)
    test_data = np.concatenate([test_data,time_features], axis=1)
    return data_t, label, test_data


def smape_lgb(y_pred, train_data):
    y_true = train_data.get_label()
    v = 2 * abs(y_pred - y_true) / (abs(y_pred) + abs(y_true))
    output = np.mean(v) * 100
    return 'smape_lgb', output, False


cols = test.columns
val_results = np.zeros((cols.shape[0], 24))
models = dict(keys=cols)
for i, col in enumerate(cols):
    home = test[col].values
    time = test_time.copy()

    # drop NaN
    first_non_nan_idx = np.where(~np.isnan(home))[0][0]
    home = home[first_non_nan_idx:]
    time = time[first_non_nan_idx:]
    cut = home.shape[0] % 24
    home = home[cut:]
    time = time[cut:]

    # transformation
    data, label, _ = trans(home, time, window=window)
    x_train, x_valid, y_train, y_valid = train_test_split(data, label, test_size=0.2, random_state=seed)

    # Train model
    models_24 = []
    for h in range(window):
        lgb_train = lgb.Dataset(x_train, label=y_train[:,h])
        lgb_valid = lgb.Dataset(x_valid, label=y_valid[:,h])
        params = {
            'random_seed': seed,
            'bagging_seed': seed,
            'feature_fraction_seed': seed,
            'data_random_seed': seed,
            'drop_seed': seed,
            'boosting_type': 'gbdt',
            'objective': 'huber',
            'learning_rate': 0.04,
            'num_leaves': 63,
            'max_depth': -1,
            'bagging_fraction': 0.1,
            'feature_fraction': 0.4,
            'lambda_l1': 10.0,
            'lambda_l2': 30.0,
            'max_bin': 255,
            'verbosity':-1,
            'n_jobs': -1
        }

        model = lgb.train(
        params,
        lgb_train,
        valid_sets=lgb_valid,
        num_boost_round=2000,
        early_stopping_rounds=100,
        verbose_eval=-1,
        feature_name=None,
        feval=smape_lgb
        )
        val_result = model.best_score['valid_0']['smape_lgb']
        models_24.append(model)
        val_results[i, h] = val_result
    print(f'For {col}, smape is {val_results[i,:].mean()}')
    models[col] = models_24

# test 예측
submission = pd.read_csv('submit/submission.csv')
submission.index = submission['meter_id']
submission.drop(columns=['meter_id'], inplace=True)
for col in cols:
    _, _, test_data = trans(home, time, window=window)
    for h in range(window):
        submission.loc[col,'X2018_7_1_'+str(h+1)+'h'] = models[col][h].predict(test_data)

#%% 일별 및 월별 예측
def AR_day_set(data,time):
    temp_day = np.zeros([6, 7])  # pre-allocation for output dataset
    mon_idx = np.zeros([1, 7])  # 몇 번째 week인지 확인하는 idx
    for ii in range(0, len(data) - 500):

        idx = len(data) - ii - 1

        day_idx = time.iloc[idx].weekday()  # data의 요일정보
        time_idx = time.iloc[idx].hour  # data의 시간정보

        if mon_idx[0, day_idx] < 6:  # 6번째 week 이상이면 추가 X

            if pd.isnull(data[idx]):  # bad data restortion
                res_data = np.zeros([1, 9])
                # 1주전, 2주전, 3주전의 같은 요일, 시간 데이터를 저장 후 mean
                res_data[0, 0] = data[idx - 24 * 7 - 1]
                res_data[0, 1] = data[idx - 24 * 7]
                res_data[0, 2] = data[idx - 24 * 7 + 1]

                res_data[0, 3] = data[idx - 48 * 7 - 1]
                res_data[0, 4] = data[idx - 48 * 7]
                res_data[0, 5] = data[idx - 48 * 7 + 1]

                res_data[0, 6] = data[idx - 1]
                res_data[0, 7] = data[idx - 3 * 24 * 7]
                res_data[0, 8] = data[idx + 1]
                # 하루 사용량 저장을 위한 시간 데이터 합
                temp_day[int(round(mon_idx[0, day_idx])), day_idx] = temp_day[int(
                    round(mon_idx[0, day_idx])), day_idx] + np.nanmean(res_data)
            else:
                # 하루 사용량 저장을 위한 시간 데이터 합
                temp_day[int(round(mon_idx[0, day_idx])), day_idx] = temp_day[
                                                                         int(round(mon_idx[0, day_idx])), day_idx] + \
                                                                     data[idx]
            if time_idx == 0:
                # 요일이 지나면, week확인 idx +1
                mon_idx[0, day_idx] = mon_idx[0, day_idx] + 1
    return temp_day

for col in cols:
    fcst_d = np.zeros([1, 10])  # pre-allocation of the result data
    home = test[col].values
    time = test_time.copy()

    # drop NaN
    first_non_nan_idx = np.where(~np.isnan(home))[0][0]
    home = home[first_non_nan_idx:]
    time = time[first_non_nan_idx:]
    cut = home.shape[0] % 24
    home = home[cut:]
    time = time[cut:]

    trainAR_Day = AR_day_set(home, time)  # 데이터를 불러옵니다.

    # Similar day aprroach
    day_idx = np.zeros([1, 10])
    for ii in range(0, 10):
        mod_idx = -1
        temp_idx = (ii + mod_idx) % 7  # 예측하는 날에 맞는 요일 idx를 정리.
        day_idx[0, ii] = temp_idx

    for ii in range(0, 10):
        flen = np.random.randint(3) + 2  # 2~5개까지 랜덤하게 과거 데이터를 불러옵니다.

        temp_day = np.zeros([1, flen])

        for jj in range(0, flen):
            temp_day[0, jj] = trainAR_Day[jj, int(round(day_idx[0, ii]))]

        # 불러온 데이터를 평균을하여 예측함.
        fcst_d[0, ii] = np.mean(temp_day)

    for i in range(10):
        submission.loc[col, 'X2018_7_' + str(i + 1) + '_d'] = fcst_d[0][i]  # column명을 submission 형태에 맞게 지정합니다.

    ### 월별 예측
    mon_test = np.zeros([1, 300])

    # Similar day aprroach
    day_idx = np.zeros([1, 300])
    for ii in range(0, 300):
        mod_idx = -1
        temp_idx = (ii + mod_idx) % 7  # 요일 idx 생성(월~일: 0~6)
        day_idx[0, ii] = temp_idx

    # 휴일의 경우, 일요일과 같은 데이터로 가정함.
    day_idx[0, 31 + 15 - 1] = 6  # 광복절
    day_idx[0, 31 + 31 + 24 - 1] = 6  # 추석
    day_idx[0, 31 + 31 + 25 - 1] = 6  # 추석
    day_idx[0, 31 + 31 + 26 - 1] = 6  # 대체휴일
    day_idx[0, 31 + 31 + 30 + 3 - 1] = 6  # 개천절
    day_idx[0, 31 + 31 + 30 + 9 - 1] = 6  # 한글날
    day_idx[0, 31 + 31 + 30 + 31 + 30 + 25 - 1] = 6  # 성탄절

    for ii in range(0, 300):
        flen = np.random.randint(3) + 1  # Similar day approach를 위한 1~4개의 데이터 추출
        temp_day = np.zeros([1, flen])
        for jj in range(0, flen):
            temp_day[0, jj] = trainAR_Day[jj, int(round(day_idx[0, ii]))]
        mon_test[0, ii] = np.mean(temp_day)

    # 결과 합
    pred_7m = np.sum(mon_test[0, 0:31])
    pred_8m = np.sum(mon_test[0, 31:62])
    pred_9m = np.sum(mon_test[0, 62:92])
    pred_10m = np.sum(mon_test[0, 92:123])
    pred_11m = np.sum(mon_test[0, 123:153])

    submission.loc[col, 'X2018_7_m'] = pred_7m  # 7월
    submission.loc[col, 'X2018_8_m'] = pred_8m  # 8월
    submission.loc[col, 'X2018_9_m'] = pred_9m  # 9월
    submission.loc[col, 'X2018_10_m'] = pred_10m  # 10월
    submission.loc[col, 'X2018_11_m'] = pred_11m  # 11월

#%% 제출
submission.to_csv('submit/submit_4.csv', index=True)

#%% 확인
a = pd.read_csv('submit/submit_2.csv')
b = pd.read_csv('submit/submit_4.csv')

smape(b.iloc[:,1:24].values, a.iloc[:,1:24].values)