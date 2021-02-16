import pandas as pd
import numpy as np
import os
from tqdm import tqdm
try:
    import cPickle as pickle
except BaseException:
    import pickle
from pathlib import Path
import tensorflow as tf


# 시드 설정 및 데이터 load
tf.compat.v1.set_random_seed(42)
np.random.seed(42)
test = pd.read_csv('data/test.csv')
test['Time'] = pd.to_datetime(test.Time)
test = test.set_index('Time')
submission = pd.read_csv('data/submission.csv')
submission.index = submission['meter_id']
submission.drop(columns=['meter_id'], inplace=True)

# load parameters
dct_param_list = pd.read_csv('data/parameters/dct_params.csv',dtype={'max_depth':int,'min_samples_leaf':int})
dct_param_list.set_index('Unnamed: 0', inplace=True)
dnn_param_list = pd.read_csv('data/parameters/dnn_params.csv',dtype={'EPOCH':int,'h1':int,'h2':int,'h3':int,'h4':int})
dnn_param_list.set_index('Unnamed: 0', inplace=True)
extra_param_list = pd.read_csv('data/parameters/extra_params.csv',dtype={'max_depth':int,'min_samples_leaf':int,'n_jobs':int})
extra_param_list.set_index('Unnamed: 0', inplace=True)
rf_param_list = pd.read_csv('data/parameters/rf_params.csv',dtype={'max_depth':int,'min_samples_leaf':int,'n_estimators':int,'random_state':int,'n_jobs':int})
rf_param_list.set_index('Unnamed: 0', inplace=True)
svr_param_list = pd.read_csv('data/parameters/svr_params.csv')
svr_param_list.set_index('Unnamed: 0', inplace=True)

#%% 함수 define
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor

def smape(true, pred):
    v = 2 * abs(pred - true) / (abs(pred) + abs(true))
    output = np.mean(v) * 100
    return output

def AR_data_set(Data, place_id, prev_type, Curr_type):
    # Mon: 0 ~ Sun:6

    TrainAR = [];
    TestAR = []
    len_bad = 20  # 하루 내 NaN의 개수 기준, (24-len_bad)보다 많으면 그 날은 제거
    Power = Data[place_id].iloc  # test.csv에서 특정 id의 전력 데이터
    Date = Data[place_id].index  # test.csv에서 특정 id의 날짜 데이터
    prev_aloc = [0] * 24;
    curr_aloc = [0] * 24  # pre-allocation

    for ii in range(24, len(Date)):

        if (Date[ii].hour == 0) & (ii > 48) & (np.sum(curr_aloc) != 24 * curr_aloc[1]) & (
                np.sum(prev_aloc) != 24 * prev_aloc[1]):
            prev_idx = 0;
            curr_idx = 0  # bad data idx

            for kk in range(0, 24):
                if prev_aloc[kk] > -20:  # check the bad data.
                    prev_idx = prev_idx + 1
                else:  # interpolate the bad data.
                    # bad data일 경우, 앞뒤로 20개의 포인트를 가져와서
                    # interpolation 진행.
                    temp = np.zeros([1, 41])
                    for qq in range(0, 41):
                        temp[0, qq] = Power[(ii - 24) - (24 - kk) - 20 + qq]

                    temp_temp = pd.DataFrame(data=temp)

                    temp = temp_temp.interpolate('spline', order=1)
                    temp = temp.values
                    prev_aloc[kk] = temp[0, 20]

            for kk in range(0, 24):
                if curr_aloc[kk] > -20:  # check the bad data.
                    curr_idx = curr_idx + 1
                else:
                    # bad data일 경우, 앞뒤로 20개의 포인트를 가져와서
                    # interpolation 진행.
                    temp = np.zeros([1, 41])
                    for qq in range(0, 41):
                        temp[0, qq] = Power[(ii) - (24 - kk) - 20 + qq]
                    temp_temp = pd.DataFrame(data=temp)

                    temp = temp_temp.interpolate('spline', order=1)
                    temp = temp.values
                    curr_aloc[kk] = temp[0, 20]

            # bad data가 특정 개수 이상이면, data set에 추가하지 않는다.
            if (prev_idx > len_bad) & (curr_idx > len_bad):
                TrainAR.append(prev_aloc)
                TestAR.append(curr_aloc)

        # 0시에 하루 데이터 초기화.
        if Date[ii].hour == 0:
            prev_aloc = [0] * 24
            curr_aloc = [0] * 24

        # 요일 데이터 확인.
        prev_day = Date[ii - 24].weekday()
        curr_day = Date[ii].weekday()

        # 요일 데이터 타입 분류
        # Workday(1) = day type<5(월~금)
        # Workday(2) = day type>4(토~일)
        if ((prev_type == 1) & (prev_day < 5)) & ((Curr_type == 2) & (curr_day > 4)):
            prev_aloc[Date[ii - 24].hour] = Power[ii - 24]
            curr_aloc[Date[ii].hour] = Power[ii]

        if ((prev_type == 1) & (prev_day < 5)) & ((Curr_type == 1) & (curr_day < 5)):
            prev_aloc[Date[ii - 24].hour] = Power[ii - 24]
            curr_aloc[Date[ii].hour] = Power[ii]

        if ((prev_type == 2) & (prev_day > 4)) & ((Curr_type == 2) & (curr_day > 4)):
            prev_aloc[Date[ii - 24].hour] = Power[ii - 24]
            curr_aloc[Date[ii].hour] = Power[ii]

        if ((prev_type == 2) & (prev_day > 4)) & ((Curr_type == 1) & (curr_day < 5)):
            prev_aloc[Date[ii - 24].hour] = Power[ii - 24]
            curr_aloc[Date[ii].hour] = Power[ii]

    TrainAR = np.array(TrainAR)
    TestAR = np.array(TestAR)
    return TrainAR, TestAR

def AR_day_set(data, place_id):
    Power = data[place_id].values  # 전력 데이터
    Date = data[place_id].index  # 요일 데이터

    temp_day = np.zeros([6, 7])  # pre-allocation for output dataset
    mon_idx = np.zeros([1, 7])  # 몇 번째 week인지 확인하는 idx

    for ii in range(0, len(Power) - 500):

        idx = len(Power) - ii - 1

        day_idx = Date[idx].weekday()  # data의 요일정보
        time_idx = Date[idx].hour  # data의 시간정보

        if mon_idx[0, day_idx] < 6:  # 6번째 week 이상이면 추가 X

            if np.isnan(Power[idx]):  # bad data restortion
                res_data = np.zeros([1, 9])
                # 1주전, 2주전, 3주전의 같은 요일, 시간 데이터를 저장 후 mean
                res_data[0, 0] = Power[idx - 24 * 7 - 1]
                res_data[0, 1] = Power[idx - 24 * 7]
                res_data[0, 2] = Power[idx - 24 * 7 + 1]

                res_data[0, 3] = Power[idx - 48 * 7 - 1]
                res_data[0, 4] = Power[idx - 48 * 7]
                res_data[0, 5] = Power[idx - 48 * 7 + 1]

                res_data[0, 6] = Power[idx - 1]
                res_data[0, 7] = Power[idx - 3 * 24 * 7]
                res_data[0, 8] = Power[idx + 1]
                # 하루 사용량 저장을 위한 시간 데이터 합
                temp_day[int(round(mon_idx[0, day_idx])), day_idx] = temp_day[int(
                    round(mon_idx[0, day_idx])), day_idx] + np.nanmean(res_data)

            else:
                # 하루 사용량 저장을 위한 시간 데이터 합
                temp_day[int(round(mon_idx[0, day_idx])), day_idx] = temp_day[
                                                                         int(round(mon_idx[0, day_idx])), day_idx] + \
                                                                     Power[idx]

            if time_idx == 0:
                # 요일이 지나면, week확인 idx +1
                mon_idx[0, day_idx] = mon_idx[0, day_idx] + 1

    return temp_day

def linear_prediction(trainAR, testAR, flen, test_data):
    len_tr = len(trainAR[0, :])  # 시간 포인트 수
    day_t = len(trainAR)
    pred = np.empty((len(trainAR), len_tr))
    fcst = np.empty((len(trainAR), len_tr))

    for j in range(0, day_t):
        if day_t > 1:
            x_ar = np.delete(trainAR[:, len_tr - flen:len_tr], (j), axis=0)
            y = np.delete(testAR, (j), axis=0)
        else:
            x_ar = trainAR[:, len_tr - flen:len_tr]
            y = testAR

        pi_x_ar = np.linalg.pinv(x_ar)
        lpc_c = np.empty((len(x_ar), flen))

        lpc_c = np.matmul(pi_x_ar, y)

        test_e = trainAR[j, :]
        test_ex = test_e[len_tr - flen:len_tr]
        pred[j, :] = np.matmul(test_ex, lpc_c)

    x_ar = trainAR[:, len_tr - flen:len_tr]
    y = testAR
    pi_x_ar = np.linalg.pinv(x_ar)
    lpc_c = np.empty((len(x_ar), flen))

    lpc_c = np.matmul(pi_x_ar, y)

    Test_AR = testAR[0:len(testAR), :]

    smape_list = np.zeros((len(pred), 1))

    for i in range(0, len(pred)):
        smape_list[i] = smape(pred[i, :], Test_AR[i, :])
        avr_smape = np.mean(smape_list)

    test_e = test_data
    test_ex = test_e[len_tr - flen:len_tr]
    fcst = np.matmul(test_ex, lpc_c)

    return avr_smape, fcst, pred

def dnn_gen_val(trainAR, testAR, x_24hrs, params=None):
    # default parameter
    if params == None:
        params = {
            'EPOCH': 80,
            'h1': 128,
            'h2': 256,
            'h3': 128,
            'h4': 0,
            'lr': 0.001,
        }

    # load data
    numData = np.size(trainAR, 0)
    numTr = int(numData * 0.8)
    Xtr = trainAR[0:numTr, :]
    Ytr = testAR[0:numTr, :]
    Xte = trainAR[numTr:numData, :]
    Yte = testAR[numTr:numData, :]
    num_tr = np.size(trainAR, 1)
    num_te = np.size(testAR, 1)

    # Build model
    model = tf.keras.Sequential()
    model.add(layers.Dense(params['h1'], activation='relu', input_shape=(num_tr,)))
    model.add(layers.Dense(params['h2'], activation='relu'))
    model.add(layers.Dense(params['h3'], activation='relu'))
    if params['h4'] != 0:
        model.add(layers.Dense(params['h4'], activation='relu'))
    model.add(layers.Dense(num_te))
    optimizer = tf.keras.optimizers.Adam(params['lr'])
    model.compile(loss='mae',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    model.fit(Xtr, Ytr, epochs=params['EPOCH'], verbose=0)

    # predict
    y_pred = model.predict(Xte)
    smape_list = np.zeros((len(y_pred), 1))
    for i in range(0, len(y_pred)):
        smape_list[i] = smape(y_pred[i, :], Yte[i, :])
    avr_smape = np.mean(smape_list)

    ypr= model.predict(x_24hrs.reshape(1,-1))

    return ypr, avr_smape, y_pred

def lstm_gen_val(trainAR, testAR, x_24hrs, params = None):
    # default parameter
    if params == None:
        params = {
            'EPOCH': 80,
            'h1': 128,
            'h2': 256,
            'lr': 0.001,
        }
    numData = np.size(trainAR, 0)
    numTr = int(numData * 0.8)
    Xtr = trainAR[0:numTr, :]
    Ytr = testAR[0:numTr, :]

    Xte = trainAR[numTr:numData, :]
    Yte = testAR[numTr:numData, :]

    num_tr = np.size(trainAR, 1)
    num_te = np.size(testAR, 1)

    Xtr = Xtr.reshape(-1, num_tr, 1)
    Xte = Xte.reshape(-1, num_te, 1)
    Ytr = Ytr.reshape(-1, num_tr)
    Yte = Yte.reshape(-1, num_te)
    train_data = tf.data.Dataset.from_tensor_slices((Xtr, Ytr))
    train_data = train_data.shuffle(num_tr, reshuffle_each_iteration=True)
    train_data = train_data.batch(num_tr)
    train_data = train_data.repeat(params['EPOCH'])
    val_data = tf.data.Dataset.from_tensor_slices((Xte, Yte))
    val_data = val_data.batch(num_te)

    def build_model():
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(params['h1'],
                                 return_sequences=False,
                                 input_shape=Xtr.shape[-2:],
                                 activation='relu'),
            layers.Dense(params['h2'], activation='relu'),
            layers.Dense(params['h1'], activation='relu'),
            layers.Dense(num_te)
        ])

        optimizer = tf.keras.optimizers.Adam(0.001)

        model.compile(loss='mae',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model

    model = build_model()

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=30,
        verbose=0,
        mode='min',
        baseline=None,
        restore_best_weights=True
    )

    history = model.fit(
        train_data, epochs=params['EPOCH'], verbose=0, validation_data=val_data, callbacks=[es])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    y_pred = model.predict(Xte)
    smape_list = np.zeros((len(y_pred), 1))

    for i in range(0, len(y_pred)):
        smape_list[i] = smape(y_pred[i, :], Yte[i, :])
    avr_smape = np.mean(smape_list)

    x_24hrs = np.reshape(x_24hrs, (-1,24,1))
    Ypr= model.predict(x_24hrs)
    return Ypr, avr_smape, y_pred

def rf_gen_val(trainAR, testAR, x_24hrs, params = None):
    Dnum = trainAR.shape[0]
    lnum = trainAR.shape[1]
    if params == None:
        params = {
            'max_depth': 2,
            'random_state': 0,
            'n_estimators': 100,
            'criterion': 'mae'
        }
    numTr = int(Dnum * 0.8)
    y_pred = np.zeros((Dnum - numTr, 24))
    # validation
    smape_list = np.zeros([Dnum, 1])
    for ii in range(0, Dnum):  # cross validation
        trainAR_temp = np.delete(trainAR, ii, axis=0)
        testAR_temp = np.delete(testAR, ii, axis=0)

        # mae 기반의 loss를 이용한 randomforest model 생성
        regr = RandomForestRegressor(**params)
        regr.fit(trainAR_temp, testAR_temp)

        x_temp = np.zeros([1, lnum])
        for kk in range(0, lnum):
            x_temp[0, kk] = trainAR[ii, kk]

        ypr = regr.predict(x_temp)
        if ii >= numTr:
            y_pred[ii - numTr, :] = ypr
        yre = np.zeros([1, lnum])
        for kk in range(0, lnum):
            yre[0, kk] = testAR[ii, kk]
        smape_list[ii] = smape(np.transpose(ypr), np.transpose(yre))
    regr = RandomForestRegressor(**params)
    regr.fit(trainAR, testAR)
    x_24hrs = np.reshape(x_24hrs, (-1, lnum))
    ypr = regr.predict(x_24hrs)
    avg_smape = np.mean(smape_list)
    return ypr, avg_smape, y_pred

def svr_gen_val(trainAR, testAR, x_24hrs, params = None):
    Dnum = trainAR.shape[0]
    lnum = trainAR.shape[1]
    if params == None:
        params = {
            'kernel': 'rbf'
        }

    numTr = int(Dnum * 0.8)
    y_pred = np.zeros((Dnum - numTr, 24))
    # validation
    smape_list = np.zeros([Dnum, 1])
    for ii in range(0, Dnum):  # cross validation
        trainAR_temp = np.delete(trainAR, ii, axis=0)
        testAR_temp = np.delete(testAR, ii, axis=0)

        regr = MultiOutputRegressor(SVR(**params))
        regr.fit(trainAR_temp, testAR_temp)

        x_temp = np.zeros([1, lnum])
        for kk in range(0, lnum):
            x_temp[0, kk] = trainAR[ii, kk]

        ypr = regr.predict(x_temp)
        if ii >= numTr:
            y_pred[ii - numTr, :] = ypr
        yre = np.zeros([1, lnum])
        for kk in range(0, lnum):
            yre[0, kk] = testAR[ii, kk]
        smape_list[ii] = smape(np.transpose(ypr), np.transpose(yre))

    # validation
    avg_smape = np.mean(smape_list)
    regr = MultiOutputRegressor(SVR(**params))
    regr.fit(trainAR, testAR)
    x_24hrs = np.reshape(x_24hrs, (-1, lnum))
    ypr = regr.predict(x_24hrs)
    return ypr, avg_smape, y_pred

def extra_gen_val(trainAR, testAR, x_24hrs, params = None):
    Dnum = trainAR.shape[0]
    lnum = trainAR.shape[1]
    if params == None:
        params = {
            'n_estimators': 100,
            'n_jobs': -1,
            'min_samples_split': 25,
        }
    numTr = int(Dnum * 0.8)
    y_pred = np.zeros((Dnum - numTr, 24))
    # validation
    smape_list = np.zeros([Dnum, 1])
    for ii in range(0, Dnum):  # cross validation
        trainAR_temp = np.delete(trainAR, ii, axis=0)
        testAR_temp = np.delete(testAR, ii, axis=0)

        regr = ExtraTreesRegressor(**params)
        regr.fit(trainAR_temp, testAR_temp)

        x_temp = np.zeros([1, lnum])
        for kk in range(0, lnum):
            x_temp[0, kk] = trainAR[ii, kk]

        ypr = regr.predict(x_temp)
        if ii >= numTr:
            y_pred[ii - numTr, :] = ypr
        yre = np.zeros([1, lnum])
        for kk in range(0, lnum):
            yre[0, kk] = testAR[ii, kk]
        smape_list[ii] = smape(np.transpose(ypr), np.transpose(yre))
    avg_smape = np.mean(smape_list)
    regr = ExtraTreesRegressor(**params)
    regr.fit(trainAR, testAR)
    x_24hrs = np.reshape(x_24hrs, (-1, lnum))
    ypr = regr.predict(x_24hrs)
    return ypr, avg_smape, y_pred

def dct_gen_val(trainAR, testAR, x_24hrs, params = None):
    Dnum = trainAR.shape[0]
    lnum = trainAR.shape[1]
    if params == None:
        params = {
            'random_state': 0,
        }

    numTr = int(Dnum * 0.8)
    y_pred = np.zeros((Dnum - numTr, 24))
    # validation
    smape_list = np.zeros([Dnum, 1])
    for ii in range(0, Dnum):  # cross validation
        trainAR_temp = np.delete(trainAR, ii, axis=0)
        testAR_temp = np.delete(testAR, ii, axis=0)

        regr = DecisionTreeRegressor(**params)
        regr.fit(trainAR_temp, testAR_temp)

        x_temp = np.zeros([1, lnum])
        for kk in range(0, lnum):
            x_temp[0, kk] = trainAR[ii, kk]

        ypr = regr.predict(x_temp)
        if ii >= numTr:
            y_pred[ii - numTr, :] = ypr
        yre = np.zeros([1, lnum])
        for kk in range(0, lnum):
            yre[0, kk] = testAR[ii, kk]
        smape_list[ii] = smape(np.transpose(ypr), np.transpose(yre))
    avg_smape = np.mean(smape_list)
    regr = DecisionTreeRegressor(**params)
    regr.fit(trainAR, testAR)
    x_24hrs = np.reshape(x_24hrs, (-1, lnum))
    ypr = regr.predict(x_24hrs)
    return ypr, avg_smape, y_pred

def AR_val(trainAR, testAR, flen, subm_24hrs):
    len_tr = len(trainAR[0, :])  # 시간 포인트 수
    day_t = len(trainAR)
    numTr = int(day_t * 0.8)
    pred = np.empty((len(trainAR), len_tr))
    fcst = np.empty((len(trainAR), len_tr))

    for j in range(0, day_t):
        if day_t > 1:
            x_ar = np.delete(trainAR[:, len_tr - flen:len_tr], (j), axis=0)
            y = np.delete(testAR, (j), axis=0)
        else:
            x_ar = trainAR[:, len_tr - flen:len_tr]
            y = testAR
        pi_x_ar = np.linalg.pinv(x_ar)
        lpc_c = np.empty((len(x_ar), flen))
        lpc_c = np.matmul(pi_x_ar, y)
        test_e = trainAR[j, :]
        test_ex = test_e[len_tr - flen:len_tr]
        pred[j, :] = np.matmul(test_ex, lpc_c)

    x_ar = trainAR[:, len_tr - flen:len_tr]
    y = testAR
    pi_x_ar = np.linalg.pinv(x_ar)
    lpc_c = np.empty((len(x_ar), flen))
    lpc_c = np.matmul(pi_x_ar, y)
    Test_AR = testAR[0:len(testAR), :]
    smape_list = np.zeros((len(pred), 1))

    for i in range(0, len(pred)):
        smape_list[i] = smape(pred[i, :], Test_AR[i, :])
        avr_smape = np.mean(smape_list)

    test_e = subm_24hrs
    test_ex = test_e[len_tr - flen:len_tr]
    fcst = np.matmul(test_ex, lpc_c)

    return fcst, avr_smape, pred[numTr:,:]

def similar_approach_val(trainAR, testAR, slen, sim_set):
    simil_smape_list = np.zeros([1, len(testAR[:, 0])])
    numTr = int(trainAR.shape[0] * 0.8)

    y_pred = [] # validation result
    for col_ii in range(0, len(testAR[:, 0])):
        simil_mean = []
        simil_temp = np.zeros([1, 24])
        simil_idx = np.zeros([1, len(testAR[:, 0])])

        for sub_col in range(0, len(testAR[:, 0])):
            simil_idx[0, sub_col] = smape(trainAR[col_ii, :], trainAR[sub_col, :])

        testAR_temp = np.delete(testAR, np.argmin(simil_idx), axis=0)
        simil_idx = np.delete(simil_idx, np.argmin(simil_idx), axis=1)

        for search_len in range(0, slen):
            simil_mean.append(testAR_temp[np.argmin(simil_idx), :])
            testAR_temp = np.delete(testAR, np.argmin(simil_idx), axis=0)
            simil_idx = np.delete(simil_idx, np.argmin(simil_idx), axis=1)

        for row_ii in range(0, 24):
            simil_temp[0, row_ii] = np.median(testAR_temp[:, row_ii])

        simil_smape_list[0, col_ii] = smape(testAR[col_ii, :], simil_temp)
        simil_smape = np.mean(simil_smape_list)
        if col_ii >= numTr:
            y_pred.append(simil_temp)
    y_pred = np.array(y_pred)
    y_pred = np.reshape(y_pred,(-1,24))

    simil_mean = []
    simil_temp = np.zeros([1, 24])
    simil_idx = np.zeros([1, len(testAR[:, 0])])
    testAR_temp = testAR

    for sub_col in range(0, len(testAR[:, 0])):
        simil_idx[0, sub_col] = smape(sim_set, trainAR[sub_col, :])

    for search_len in range(0, slen):
        simil_mean.append(testAR_temp[np.argmin(simil_idx), :])
        testAR_temp = np.delete(testAR, np.argmin(simil_idx), axis=0)
        simil_idx = np.delete(simil_idx, np.argmin(simil_idx), axis=1)

    for row_ii in range(0, 24):
        simil_temp[0, row_ii] = np.median(testAR_temp[:, row_ii])

    return simil_temp, simil_smape, y_pred


#%% validation, test 예측 결과 저장
## 결과 저장
test_reals_all = dict()
val_preds_all = dict()
test_preds_all = dict()
smapes_all = dict()

for key_idx, key in tqdm(enumerate(test.columns)):
    prev_type = 2  # 전날 요일 타입
    curr_type = 2  # 예측날 요일 타입
    trainAR, testAR = AR_data_set(test, key, prev_type, curr_type)

    # [시간 예측을 위한 마지막 24pnt 추출]
    # NaN 값처리를 위해서 마지막 40pnts 추출 한 후에
    # interpolation하고 나서 24pnts 재추출
    temp_test = test[key]
    temp_test = test[key].iloc[8759 - 40:]
    temp_test = temp_test.interpolate(method='spline', order=2)

    temp_test = np.array(temp_test.values)
    temp_test = temp_test[len(temp_test) - 24:len(temp_test) + 1]
    subm_24hrs = temp_test
    del temp_test

    fchk = 1  # filter length
    temp_idx = []
    smape_lin = []

    # 한 행씩 linear prediction을 테스트해보고 NaN이 발견된다면, 그 행을 제거.
    for chk_bad in range(0, len(trainAR[:, 0])):
        prev_smape = 200  # SMAPE 기준값
        nan_chk = 0  # NaN chk idx

        trainAR_temp = np.zeros([1, 24])  # pre-allocation
        testAR_temp = np.zeros([1, 24])  # pre-allocation

        # 한 행씩 테스트를 하기 위한 변수 설정
        for ii in range(0, 24):
            trainAR_temp[0, ii] = trainAR[chk_bad, ii]
            testAR_temp[0, ii] = testAR[chk_bad, ii]

        # linear prediction test
        lin_sampe, fcst_temp, pred_hyb = linear_prediction(trainAR_temp, testAR_temp, fchk, subm_24hrs)

        if np.isnan(lin_sampe):  # SMAPE가 NaN 경우, 그 행을 제거
            nan_chk = 1
        if np.isnan(np.sum(trainAR_temp)):  # chk_bad의 행이 NaN을 포함할 경우 제거
            nan_chk = 1
        if np.isnan(np.sum(testAR_temp)):  # chk_bad의 행이 NaN을 포함할 경우 제거
            nan_chk = 1
        if nan_chk == 1:  # NaN 값이 있는 행 넘버를 append
            temp_idx.append(chk_bad)

    # NaN 값이 나타난 data set은 제거
    trainAR = np.delete(trainAR, temp_idx, axis=0)
    testAR = np.delete(testAR, temp_idx, axis=0)

    del_smape = np.zeros([1, len(trainAR[:, 1])])
    prev_smape = 200
    fchk = 0

    # filter length 최적화
    for chk in range(3, 24):
        # filter length을 바꿔가며 Smape가 최소가 되는 값을 찾아감.
        lin_sampe, fcst_temp, pred_hyb = linear_prediction(trainAR, testAR, chk, subm_24hrs)
        if prev_smape > lin_sampe:
            fchk = chk
            prev_smape = lin_sampe

    # 필요없는 데이터 제거
    # 한 줄(하루)씩 제거해가면서 SMAPE 결과를 분석.
    for chk_lin in range(0, len(trainAR[:, 1])):
        trainAR_temp = np.delete(trainAR, chk_lin, axis=0)
        testAR_temp = np.delete(testAR, chk_lin, axis=0)
        lin_sampe, fcst_temp, pred_hyb = linear_prediction(trainAR_temp, testAR_temp, fchk, subm_24hrs)

        del_smape[0, chk_lin] = lin_sampe

    # SMAPE에 악영향을 주는 행을 제거
    trainAR = np.delete(trainAR, np.argmin(del_smape), axis=0)
    testAR = np.delete(testAR, np.argmin(del_smape), axis=0)
    del_smape = np.delete(del_smape, np.argmin(del_smape), axis=1)

    del nan_chk, lin_sampe, fcst_temp, pred_hyb, prev_smape, temp_idx

    ## 결과 저장
    smape_results = dict()
    val_preds = dict()
    test_preds = dict()

    ## baseline results
    # AR
    fcst, smape_val, y_pred = AR_val(trainAR, testAR, fchk, subm_24hrs)
    smape_results['AR'], val_preds['AR'],test_preds['AR'] = smape_val, y_pred, fcst

    # Similar day
    temp_24hrs = np.zeros([1, 24])  # np.array type으로 변경.
    for qq in range(0, 24):
        temp_24hrs[0, qq] = subm_24hrs[qq]
    prev_smape = 200
    fsim = 0  # N개의 날
    for sim_len in range(2, 5):
        sim_fcst, sim_smape, y_pred = similar_approach_val(trainAR, testAR, sim_len, temp_24hrs)
        if prev_smape > sim_smape:
            fsim = sim_len
            prev_smape = sim_smape
    fcst, smape_val, y_pred = similar_approach_val(trainAR, testAR, fsim, temp_24hrs)
    smape_results['Sim'], val_preds['Sim'], test_preds['Sim'] = smape_val, y_pred, fcst

    ## load tuning results
    dct_param = dct_param_list.loc[key,:]
    dnn_param = dnn_param_list.loc[key,:]
    extra_param = extra_param_list.loc[key,:]
    rf_param = rf_param_list.loc[key,:]
    svr_param = svr_param_list.loc[key,:]
    param_list = [dct_param, dnn_param, extra_param, rf_param, svr_param]
    method_list = ['dct','dnn','extra','rf','svr']

    # predict
    Dnum = trainAR.shape[0]
    numTr = int(Dnum * 0.8)
    trues = testAR[numTr:,:]
    methods = []
    for method_idx in range(5):
        params = param_list[method_idx]
        method = method_list[method_idx]
        params = params.to_dict()
        if method == 'dnn': # dnn일 경우 dnn과 lstm 중 성능이 좋은 모델을 선택
            fcst_list, smape_list, y_pred_list = [], [], []
            for i in range(5):
                # params['EPOCH'] = int(params['EPOCH'])
                params = None
                fcst, smape_val, y_pred = dnn_gen_val(trainAR, testAR, subm_24hrs, params=params)
                # lstm
                # if i == 10:
                #     fcst, smape_val, y_pred = lstm_gen_val(trainAR, testAR, subm_24hrs, params = {'EPOCH': 80, 'h1': 432, 'h2': 168, 'lr': 0.0004})
                fcst_list.append(fcst)
                smape_list.append(smape_val)
                y_pred_list.append(y_pred)
                tf.keras.backend.clear_session()
            best_idx = np.argmin(smape_list, axis=0)
            smape_val = smape_list[best_idx]
            fcst = fcst_list[best_idx]
            y_pred = y_pred_list[best_idx]
        elif method == 'svr':
            params = None
            fcst, smape_val, y_pred = svr_gen_val(trainAR, testAR, subm_24hrs, params=params)
        elif method == 'rf':
            params = None
            fcst, smape_val, y_pred = rf_gen_val(trainAR, testAR, subm_24hrs, params=params)
        elif method == 'dct':
            params = None
            fcst, smape_val, y_pred = dct_gen_val(trainAR, testAR, subm_24hrs, params=params)
        elif method == 'extra':
            params = None
            fcst, smape_val, y_pred = extra_gen_val(trainAR, testAR, subm_24hrs, params=params)
        # save results
        key_name = method
        smape_results[key_name], val_preds[key_name], test_preds[key_name] = smape_val, y_pred, fcst

    test_reals_all[key] = testAR[numTr:,:]
    val_preds_all[key] = val_preds
    test_preds_all[key] = test_preds
    smapes_all[key] = smape_results
    print(f'smape result is {smape_results}')


#%% 24시간의 결과 앙상블
weight_arr = []
for key_idx, key in enumerate(test.columns):
    # load results
    test_preds = test_preds_all[key]
    true = test_reals_all[key]
    val_preds = val_preds_all[key]
    num_methods = len(val_preds)

    ## weight 결정
    methods = np.array(list(smapes_all[key].keys()))
    loss = np.array(list(smapes_all[key].values()))
    drop_idx = loss > np.min(loss) * 1.1
    weight = np.ones((num_methods))
    weight[drop_idx] = 0
    weight[loss < np.min(loss) * 1.05] = 3
    weight[np.argmin(loss)] = 5
    weight /= weight.sum()
    weight_arr.append(weight)
    ## test prediction based on weight
    ensemble_pred = np.zeros((1, 24))
    for j in range(num_methods):
        ensemble_pred += weight[j] * test_preds[methods[j]]

    ## submission
    submission.loc[key, submission.columns[0]:submission.columns[23]] = np.ravel(ensemble_pred)