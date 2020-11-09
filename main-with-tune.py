import pandas as pd  # 데이터 전처리
import numpy as np  # 데이터 전처리
import os
from util_saint import *
from tqdm import tqdm
try:
    import cPickle as pickle
except BaseException:
    import pickle
from pathlib import Path

def load_obj(name):
    with open('tune_results/' + name + '.pkl', 'rb') as f:
        trials = pickle.load(f)
        trials = sorted(trials, key=lambda k: k['loss'])
        return trials


#%%
test = pd.read_csv('data/test.csv')
submission = pd.read_csv('submit/sub_baseline.csv')

submission.index = submission['meter_id']
submission.drop(columns=['meter_id'], inplace=True)

test['Time'] = pd.to_datetime(test.Time)
test = test.set_index('Time')

print('Section [1]: Loading data...............')

#%%
for key_idx, key in tqdm(enumerate(test.columns)):
    tune_val_result = None

    with open('data_pr/' + key + '.pkl', 'rb') as f:
        data_pr = pickle.load(f)
    trainAR, testAR, subm_24hrs, fchk = data_pr

    baseline_val_result = pd.read_csv('val_results/baseline_result.csv')
    lstm_model, lstm_smape = lstm_gen(trainAR, testAR, EPOCHS=80)

    ### 튜닝 파라미터 load..
    for method in ['dnn', 'dnn_2','dnn_3', 'svr', 'svr_2', 'dct', 'extra', 'rf']:
        my_file = Path(f"tune_results/{key_idx}_{method}.pkl")
        if my_file.exists():
            if tune_val_result == None:
                tune_val_result = load_obj(f'{key_idx}_{method}')
            else:
                tune_val_result += load_obj(f'{key_idx}_{method}')

    # sorting all the results
    tune_val_result = sorted(tune_val_result, key=lambda k: k['loss'])

    # best만 남김
    tune_val_result = tune_val_result[0]

    # 기존 방법보다 smape가 낮을 때 replace
    if tune_val_result['loss'] < baseline_val_result['min_smape'].values[key_idx]:
        prev_loss = baseline_val_result['min_smape'].values[key_idx]
        improved_loss = tune_val_result['loss']
        params = tune_val_result['params']
        method = tune_val_result['method']
        print(f'For {method}, prev loss: {prev_loss} now: {improved_loss}')
        if 'dnn' in method:
            # 선택된 모델이 dnn일 시 앙상블
            ens_results = []
            for iter in range(5):
                if method == 'dnn':
                    Non_NNmodel, non_smape = non_linear_model_gen_v1(trainAR, testAR, params=params)
                elif method == 'dnn_2':
                    Non_NNmodel, non_smape = non_linear_model_gen_v2(trainAR, testAR, params=params)
                elif method == 'dnn_3':
                    Non_NNmodel, non_smape = non_linear_model_gen_v3(trainAR, testAR, params=params)
                temp_24hrs = np.zeros([1, 24])
                for qq in range(0, 24):
                    temp_24hrs[0, qq] = subm_24hrs[qq]
                fcst = Non_NNmodel.predict(temp_24hrs)
                ens_results.append(fcst)
            fcst = np.mean(ens_results, axis=0)
        elif method == 'svr':
            fcst, svr_smape = svr_gen(trainAR, testAR, subm_24hrs, params = params)
        elif method == 'rf':
            fcst, Mac_smape = machine_learn_gen(trainAR, testAR, subm_24hrs, params = params)
        elif method == 'dct':
            fcst, dct_smape = dct_gen(trainAR, testAR, subm_24hrs, params = params)
        elif method == 'extra':
            fcst, extra_smape = extra_gen(trainAR, testAR, subm_24hrs, params = params)
        submission.loc[key, 0:24] = np.ravel(fcst)
    else:
        print('Using baseline model of '+baseline_val_result['selected_model'].values[key_idx])


#%% save new result
submission.to_csv('submit/submit_11.csv',index=True)