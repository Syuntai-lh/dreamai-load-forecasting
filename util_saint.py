import pandas as pd  # 데이터 전처리
import numpy as np  # 데이터 전처리
import os
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def smape(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


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

def similar_approach(trainAR, testAR, slen, sim_set):
    simil_smape_list = np.zeros([1, len(testAR[:, 0])])

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

    return simil_smape, simil_temp

def machine_learn_gen(trainAR, testAR, x_24hrs, params=None):
    Dnum = trainAR.shape[0]
    lnum = trainAR.shape[1]
    smape_list = np.zeros([Dnum, 1])
    if params == None:
        params = {
            'max_depth' : 5,
            'random_state' : 0,
            'n_estimators' : 100,
            'criterion' : 'mae'
        }

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

        yre = np.zeros([1, lnum])
        for kk in range(0, lnum):
            yre[0, kk] = testAR[ii, kk]

        smape_list[ii] = smape(np.transpose(ypr), np.transpose(yre))

    regr = RandomForestRegressor(**params)
    regr.fit(trainAR, testAR)

    x_24hrs = np.reshape(x_24hrs, (-1, lnum))

    avr_smape = np.mean(smape_list)
    ypr = regr.predict(x_24hrs)

    return ypr, avr_smape

def svr_gen(trainAR, testAR, x_24hrs, params = None):
    Dnum = trainAR.shape[0]
    lnum = trainAR.shape[1]
    smape_list = np.zeros([Dnum, 1])
    if params == None:
        params = {
            'kernel': 'rbf',
        }
    for ii in range(0, Dnum):  # cross validation
        trainAR_temp = np.delete(trainAR, ii, axis=0)
        testAR_temp = np.delete(testAR, ii, axis=0)

        # mae 기반의 loss를 이용한 randomforest model 생성
        regr = MultiOutputRegressor(SVR(**params))
        regr.fit(trainAR_temp, testAR_temp)
        x_temp = np.zeros([1, lnum])
        for kk in range(0, lnum):
            x_temp[0, kk] = trainAR[ii, kk]

        ypr = regr.predict(x_temp)

        yre = np.zeros([1, lnum])
        for kk in range(0, lnum):
            yre[0, kk] = testAR[ii, kk]

        smape_list[ii] = smape(np.transpose(ypr), np.transpose(yre))

    regr = MultiOutputRegressor(SVR(**params))
    regr.fit(trainAR, testAR)

    x_24hrs = np.reshape(x_24hrs, (-1, lnum))

    avr_smape = np.mean(smape_list)
    ypr = regr.predict(x_24hrs)

    return ypr, avr_smape

def catboost_gen(trainAR, testAR, x_24hrs):
    Dnum = trainAR.shape[0]
    lnum = trainAR.shape[1]
    smape_list = np.zeros([Dnum, 1])

    for ii in range(0, Dnum):  # cross validation
        trainAR_temp = np.delete(trainAR, ii, axis=0)
        testAR_temp = np.delete(testAR, ii, axis=0)

        regr = MultiOutputRegressor(CatBoostRegressor(iterations=1000,
                                                      learning_rate=0.1,
                                                      depth=5, n_jobs=-1), n_jobs=-1)

        regr.fit(trainAR_temp, testAR_temp)
        x_temp = np.zeros([1, lnum])
        for kk in range(0, lnum):
            x_temp[0, kk] = trainAR[ii, kk]

        ypr = regr.predict(x_temp)

        yre = np.zeros([1, lnum])
        for kk in range(0, lnum):
            yre[0, kk] = testAR[ii, kk]

        smape_list[ii] = smape(np.transpose(ypr), np.transpose(yre))

    regr = MultiOutputRegressor(CatBoostRegressor(iterations=1000,
                             learning_rate=0.1,
                             depth=5), n_jobs=-1)
    regr.fit(trainAR, testAR)
    x_24hrs = np.reshape(x_24hrs, (-1, lnum))
    avr_smape = np.mean(smape_list)
    ypr = regr.predict(x_24hrs)

    return ypr, avr_smape

def dct_gen(trainAR, testAR, x_24hrs, params = None):
    Dnum = trainAR.shape[0]
    lnum = trainAR.shape[1]
    smape_list = np.zeros([Dnum, 1])
    if params == None:
        params = {
           'random_state' : 0,
        }
    for ii in range(0, Dnum):  # cross validation
        trainAR_temp = np.delete(trainAR, ii, axis=0)
        testAR_temp = np.delete(testAR, ii, axis=0)

        regr = DecisionTreeRegressor(**params)

        regr.fit(trainAR_temp, testAR_temp)
        x_temp = np.zeros([1, lnum])
        for kk in range(0, lnum):
            x_temp[0, kk] = trainAR[ii, kk]

        ypr = regr.predict(x_temp)

        yre = np.zeros([1, lnum])
        for kk in range(0, lnum):
            yre[0, kk] = testAR[ii, kk]

        smape_list[ii] = smape(np.transpose(ypr), np.transpose(yre))

    regr = DecisionTreeRegressor(**params)
    regr.fit(trainAR, testAR)
    x_24hrs = np.reshape(x_24hrs, (-1, lnum))
    avr_smape = np.mean(smape_list)
    ypr = regr.predict(x_24hrs)

    return ypr, avr_smape
def extra_gen(trainAR, testAR, x_24hrs,  params = None):
    Dnum = trainAR.shape[0]
    lnum = trainAR.shape[1]
    smape_list = np.zeros([Dnum, 1])
    if params == None:
        params = {
            'n_estimators' : 100,
            'n_jobs' : -1,
            'min_samples_split' : 25,
        }
    for ii in range(0, Dnum):  # cross validation
        trainAR_temp = np.delete(trainAR, ii, axis=0)
        testAR_temp = np.delete(testAR, ii, axis=0)

        regr = ExtraTreesRegressor(**params)

        regr.fit(trainAR_temp, testAR_temp)
        x_temp = np.zeros([1, lnum])
        for kk in range(0, lnum):
            x_temp[0, kk] = trainAR[ii, kk]

        ypr = regr.predict(x_temp)

        yre = np.zeros([1, lnum])
        for kk in range(0, lnum):
            yre[0, kk] = testAR[ii, kk]

        smape_list[ii] = smape(np.transpose(ypr), np.transpose(yre))

    regr = ExtraTreesRegressor(**params)
    regr.fit(trainAR, testAR)
    x_24hrs = np.reshape(x_24hrs, (-1, lnum))
    avr_smape = np.mean(smape_list)
    ypr = regr.predict(x_24hrs)

    return ypr, avr_smape

def light_gbm_learn_gen(trainAR, testAR, x_24hrs):
    Dnum = trainAR.shape[0]
    lnum = trainAR.shape[1]
    smape_list = np.zeros([Dnum, 1])

    def smape_lgb(y_pred, train_data):
        y_true = train_data.get_label()
        v = 2 * abs(y_pred - y_true) / (abs(y_pred) + abs(y_true))
        output = np.mean(v) * 100
        return 'smape_lgb', output, False

    for ii in range(0, Dnum):  # cross validation
        trainAR_temp = np.delete(trainAR, ii, axis=0)
        testAR_temp = np.delete(testAR, ii, axis=0)

        # lightgbm model 생성
        """
        lgb_params = {
            'boosting_type':'gbdt',
            'objective':'regression',
            'early_stopping':50,
            'num_iteration':10000,
            'num_leaves':31,
            'is_enable_sparse':'true',
            'tree_learner':'data',
            'min_data_in_leaf':600,
            'max_depth':4, 
            'learning_rate':0.1, 
            'n_estimators':675, 
            'max_bin':255, 
            'subsample_for_bin':50000, 
            'min_split_gain':5, 
            'min_child_weight':5, 
            'min_child_samples':10, 
            'subsample':0.995, 
            'subsample_freq':1, 
            'colsample_bytree':1, 
            'reg_alpha':0, 
            'reg_lambda':0, 
            'seed':0, 
            'nthread':-1, 
            'silent':True,
        }
        """
        seed = 777
        params = {
            'random_seed': seed,
            'bagging_seed': seed,
            'feature_fraction_seed': seed,
            'data_random_seed': seed,
            'drop_seed': seed,
            'boosting_type': 'gbdt',
            'objective': 'huber',
            'verbosity': -1,
            'n_jobs': -1,
        }
        x_temp = np.zeros([1, lnum])
        for kk in range(0, lnum):
            x_temp[0, kk] = trainAR[ii, kk]

        yre = np.zeros([1, lnum])
        for kk in range(0, lnum):
            yre[0, kk] = testAR[ii, kk]

        results_24 = np.zeros(24)
        ypr = np.zeros(24)
        for h in range(24):
            lgb_train = lgb.Dataset(trainAR_temp, label=testAR_temp[:, h])
            lgb_valid = lgb.Dataset(x_temp, label=yre[:, h])
            model = lgb.train(
                params,
                lgb_train,
                valid_sets=lgb_valid,
                num_boost_round=500,
                early_stopping_rounds=50,
                verbose_eval=-1,
                feature_name=None,
                feval=smape_lgb
            )
            ypr[h] = model.predict(x_temp)

        smape_list[ii] = smape(ypr, yre)

    # fit and predict
    x_24hrs = np.reshape(x_24hrs, (-1, lnum))
    ypr = np.zeros(24)
    for h in range(24):
        lgb_train = lgb.Dataset(trainAR, label=testAR[:, h])
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            verbose_eval=-1,
            feature_name=None,
            feval=smape_lgb
        )
        ypr[h] = model.predict(x_24hrs)
    avr_smape = np.mean(smape_list)
    return ypr, avr_smape

def non_linear_model_gen_v1(trainAR, testAR, params):
    if params == None:
        params = {
            'EPOCH': 80,
            'h1': 128,
            'h2': 256,
            'h3': 128,
            'lr':0.001
        }

    numData = np.size(trainAR, 0)
    numTr = int(numData * 0.8)
    Xtr = trainAR[0:numTr, :]
    Ytr = testAR[0:numTr, :]

    Xte = trainAR[numTr:numData, :]
    Yte = testAR[numTr:numData, :]

    num_tr = np.size(trainAR, 1)
    num_te = np.size(testAR, 1)

    def build_model():
        model = keras.Sequential([
            layers.Dense(params['h1'], activation='relu', input_shape=(num_tr,)),
            layers.Dense(params['h2'], activation='relu'),
            layers.Dense(params['h1'], activation='relu'),
            layers.Dense(num_te)
        ])

        optimizer = tf.keras.optimizers.Adam(params['lr'])

        model.compile(loss='mae',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model

    model = build_model()

    #    model.summary()

    # example_batch = Xtr[:10]
    # example_result = model.predict(example_batch)
    # example_result

    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')

    history = model.fit(
        Xtr, Ytr,
        epochs=params['EPOCH'], verbose=0,
        callbacks=[PrintDot()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    Ypr = model.predict(Xte)

    smape_list = np.zeros((len(Ypr), 1))

    for i in range(0, len(Ypr)):
        smape_list[i] = smape(Ypr[i, :], Yte[i, :])
    avr_smape = np.mean(smape_list)

    return model, avr_smape

def non_linear_model_gen_v3(trainAR, testAR, params):
    if params == None:
        params = {
            'EPOCH': 80,
            'h1': 128,
            'h2': 256,
            'h3': 128,
            'h4': 0,
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

    # Build model
    model = tf.keras.Sequential()
    model.add(layers.Dense(params['h1'], activation='relu', input_shape=(num_tr,)))
    model.add(layers.Dense(params['h2'], activation='relu'))
    model.add(layers.Dense(params['h2'], activation='relu'))
    model.add(layers.Dense(params['h1'], activation='relu'))
    model.add(layers.Dense(num_te))

    optimizer = tf.keras.optimizers.Adam(params['lr'])

    model.compile(loss='mae',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    #    model.summary()

    # example_batch = Xtr[:10]
    # example_result = model.predict(example_batch)
    # example_result

    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')

    history = model.fit(
        Xtr, Ytr,
        epochs=params['EPOCH'], verbose=0,
        callbacks=[PrintDot()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    Ypr = model.predict(Xte)

    smape_list = np.zeros((len(Ypr), 1))

    for i in range(0, len(Ypr)):
        smape_list[i] = smape(Ypr[i, :], Yte[i, :])
    avr_smape = np.mean(smape_list)

    return model, avr_smape

def non_linear_model_gen_v2(trainAR, testAR, params):
    if params == None:
        params = {
            'EPOCH': 80,
            'h1': 128,
            'h2': 256,
            'h3': 128,
            'h4': 0,
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

    # Build model
    model = tf.keras.Sequential()
    model.add(layers.Dense(params['h1'], activation='relu', input_shape=(num_tr,)))
    model.add(layers.Dense(params['h2'], activation='relu'))
    model.add(layers.Dense(params['h3'], activation='relu'))
    model.add(layers.Dense(params['h4'], activation='relu'))
    model.add(layers.Dense(num_te))

    optimizer = tf.keras.optimizers.Adam(params['lr'])

    model.compile(loss='mae',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    #    model.summary()

    # example_batch = Xtr[:10]
    # example_result = model.predict(example_batch)
    # example_result

    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')

    history = model.fit(
        Xtr, Ytr,
        epochs=params['EPOCH'], verbose=0,
        callbacks=[PrintDot()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    Ypr = model.predict(Xte)

    smape_list = np.zeros((len(Ypr), 1))

    for i in range(0, len(Ypr)):
        smape_list[i] = smape(Ypr[i, :], Yte[i, :])
    avr_smape = np.mean(smape_list)

    return model, avr_smape


def lstm_gen(trainAR, testAR, EPOCHS):
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
    train_data = train_data.repeat(EPOCHS)
    val_data = tf.data.Dataset.from_tensor_slices((Xte, Yte))
    val_data = val_data.batch(num_te)

    def build_model():
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128,
                                 return_sequences=False,
                                 input_shape=Xtr.shape[-2:],
                                 activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_te)
        ])

        optimizer = tf.keras.optimizers.Adam(0.001)

        model.compile(loss='mae',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model

    model = build_model()

    es = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=30,
        verbose=0,
        mode='min',
        baseline=None,
        restore_best_weights=True
    )

    history = model.fit(
        train_data, epochs=EPOCHS, verbose=1, validation_data=val_data, callbacks=[es])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    Ypr = model.predict(Xte)
    smape_list = np.zeros((len(Ypr), 1))

    for i in range(0, len(Ypr)):
        smape_list[i] = smape(Ypr[i, :], Yte[i, :])
    avr_smape = np.mean(smape_list)
    return model, avr_smape