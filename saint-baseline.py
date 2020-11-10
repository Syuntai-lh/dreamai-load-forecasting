import pandas as pd  # 데이터 전처리
import numpy as np  # 데이터 전처리
import os
from util_saint import *
from tqdm import tqdm
try:
    import cPickle as pickle
except BaseException:
    import pickle

# save_dir = 'val_results/1107_'
save_dir = None

#%%

test = pd.read_csv('data/test.csv')
submission = pd.read_csv('submit/sub_baseline.csv')

test['Time'] = pd.to_datetime(test.Time)
test = test.set_index('Time')

print('Section [1]: Loading data...............')
"""
for key in tqdm(test.columns):
    prev_type = 2  # 전날 요일 타입
    curr_type = 2  # 예측날 요일 타입
    trainAR, testAR = AR_data_set(test, key, prev_type, curr_type)

    print('Section [2]: Data generation for training set...............')

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

    print('Section [3]: mitigating bad data...............')

    del nan_chk, lin_sampe, fcst_temp, pred_hyb, prev_smape, temp_idx

    data_pr = [trainAR, testAR, subm_24hrs, fchk]
    with open('data_pr/' + key + '.pkl', 'wb') as f:
        pickle.dump(data_pr, f, pickle.HIGHEST_PROTOCOL)
"""
#%% model
comp_smape = []
for key in tqdm(test.columns):
    with open('data_pr/' + key + '.pkl', 'rb') as f:
        data_pr = pickle.load(f)
    trainAR, testAR, subm_24hrs, fchk = data_pr
    # DNN model
    Non_NNmodel, non_smape = non_linear_model_gen_v1(trainAR, testAR)

    # random forest model
    mac_fcst, Mac_smape = machine_learn_gen(trainAR, testAR, subm_24hrs)

    # svr model
    svr_fcst, svr_smape = svr_gen(trainAR, testAR, subm_24hrs)

    # linear model
    lin_sampe, fcst_temp, pred_hyb = linear_prediction(trainAR, testAR, fchk, subm_24hrs)

    # dct model
    dct_fcst, dct_smape = dct_gen(trainAR, testAR, subm_24hrs)

    # extra model
    extra_fcst, extra_smape = extra_gen(trainAR, testAR, subm_24hrs)

    # LSTM model
    lstm_model, lstm_smape = lstm_gen(trainAR, testAR)

    # Similar day approach model
    temp_24hrs = np.zeros([1, 24])  # np.array type으로 변경.
    for qq in range(0, 24):
        temp_24hrs[0, qq] = subm_24hrs[qq]

    # Similar day approach model 최적화 (몇 개의 날(N)을 가져오는 게 좋은 지 평가.)
    prev_smape = 200
    fsim = 0  # N개의 날
    for sim_len in range(2, 5):
        sim_smape, sim_fcst = similar_approach(trainAR, testAR, sim_len, temp_24hrs)
        if prev_smape > sim_smape:
            fsim = sim_len
            prev_smape = sim_smape

    # Similar day approach model
    sim_smape, sim_fcst = similar_approach(trainAR, testAR, fsim, temp_24hrs)
    # ---------------------------------------------------------------------------------------

    minor_idx = 0  # Autoregression model에서 minor value가 나타나면,
    # 모델을 Autoregression model에서 similar day appreach로 변경 진행.

    ##### Model Selection #####

    smape_results = [non_smape, lin_sampe, Mac_smape,
                     sim_smape ,svr_smape, dct_smape, extra_smape, lstm_smape]

    test_pred = pd.DataFrame()
    ## DNN
    temp_24hrs = np.zeros([1, 24])
    for qq in range(0, 24):
        temp_24hrs[0, qq] = subm_24hrs[qq]
    dnn_fcst = Non_NNmodel.predict(temp_24hrs)
    test_pred[key+'_dnn'] = np.ravel(dnn_fcst)
    ## AR
    fcst = fcst_temp
    test_pred[key+'_AR'] = np.ravel(fcst)
    ## RF
    test_pred[key+'_RF'] = np.ravel(mac_fcst)
    ## Sim
    test_pred[key+'_sim'] = np.ravel(sim_fcst)
    ## svr
    test_pred[key+'_svr'] = np.ravel(svr_fcst)
    ## extra
    test_pred[key+'_extra'] = np.ravel(extra_fcst)
    ## lstm
    temp_24hrs = np.zeros([1, 24])
    for qq in range(0, 24):
        temp_24hrs[0, qq] = subm_24hrs[qq]
    temp_24hrs = np.reshape(temp_24hrs, (-1,24,1))
    lstm_fcst = lstm_model.predict(temp_24hrs)
    test_pred[key+'_lstm'] = np.ravel(lstm_fcst)


# 각 SMAPE 결과 값을 정
comp_smape.append(smape_results)
comp_smape = np.array(comp_smape)
models = ['non', 'lin', 'Mac', 'sim', 'svr', ' extra', 'lstm']
smape_result = pd.DataFrame(index=test.columns, data=comp_smape, columns=models)

if save_dir == None:
    pass
else:
    test_pred.to_csv(save_dir+'pred_result.csv', index=True)
    smape_result.to_csv(save_dir+'smape_result.csv', index=True)
