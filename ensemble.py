import pandas as pd
import numpy as np
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
tf.compat.v1.set_random_seed(42)
np.random.seed(42)

#%% load result
tune_val_result = None
submission = pd.read_csv('submit/sub_baseline_with_lstm.csv')
submission.index = submission['meter_id']
submission.drop(columns=['meter_id'], inplace=True)
val_smape = pd.read_csv('val_results/val_smape_v2.csv')
val_smape.index = val_smape.iloc[:,0]
val_smape = val_smape.iloc[:,1:]
val_smape['baseline_min_loss'] = np.min(val_smape.values, axis=1)
val_smape['baseline_min_method'] = [val_smape.columns[:5][idx] for idx in np.argmin(val_smape.values, axis=1)]

#%% validation의 smape를 합침
val_smape['tuning_1_loss'] = [''] * 200
val_smape['tuning_1_method'] = [''] * 200
val_smape['tuning_2_loss'] = [''] * 200
val_smape['tuning_2_method'] = [''] * 200

for key_idx, key in tqdm(enumerate(val_smape.index)):
    selected_val_result = []
    for method in ['dnn', 'svr', 'dct', 'extra', 'rf']:
        my_file = Path(f"tune_results/{key_idx}_{method}.pkl")
        if my_file.exists():
            tune_val_result = load_obj(f'{key_idx}_{method}')
            selected_val_result.append(tune_val_result[0])

    # sorting all the results
    selected_val_result = sorted(selected_val_result, key=lambda k: k['loss'])

    # best loss 2개 남김
    result_1, result_2 = selected_val_result[0], selected_val_result[1]
    break_idx = 2
    # 같은 모델은 선택 안함
    while True:
        if result_1['method'] == result_2['method']:
            result_2 = selected_val_result[break_idx]
            break_idx += 1
        else:
            break
    val_smape.loc[key, 'tuning_1_method'] = result_1['method']
    val_smape.loc[key, 'tuning_1_loss'] = result_1['loss']
    val_smape.loc[key, 'tuning_2_method'] = result_2['method']
    val_smape.loc[key, 'tuning_2_loss'] = result_2['loss']

    #### weight 결정
    loss = [val_smape.loc[key, 'baseline_min_loss'], val_smape.loc[key, 'tuning_1_loss'], val_smape.loc[key, 'tuning_2_loss']]
    loss = np.array(loss)
    drop_idx = loss > np.min(loss) * 1.1
    weight = np.ones((3))
    weight[drop_idx] = 0
    weight[loss < np.min(loss) * 1.05] = 3
    weight[np.argmin(loss)] = 5
    weight /= weight.sum()

    #### predict
    with open('data_pr/' + key + '.pkl', 'rb') as f:
        data_pr = pickle.load(f)
    trainAR, testAR, subm_24hrs, fchk = data_pr
    preds = []
    preds.append(np.ravel(submission.loc[key,submission.columns[0]:submission.columns[23]].values))
    for tmp_var in [result_1, result_2]:
        params = tmp_var['params']
        method = tmp_var['method']
        if method == 'dnn':
            ens_results = []
            # 선택된 모델이 dnn일 시 3번 앙상블
            for iter in range(3):
                Non_NNmodel, non_smape = non_linear_model_gen(trainAR, testAR, params=params)
                temp_24hrs = np.zeros([1, 24])
                for qq in range(0, 24):
                    temp_24hrs[0, qq] = subm_24hrs[qq]
                fcst = Non_NNmodel.predict(temp_24hrs)
                ens_results.append(fcst)
            fcst = np.mean(ens_results, axis=0)
        elif method == 'svr':
            fcst, svr_smape = svr_gen(trainAR, testAR, subm_24hrs, params=params)
        elif method == 'rf':
            fcst, Mac_smape = machine_learn_gen(trainAR, testAR, subm_24hrs, params=params)
        elif method == 'dct':
            fcst, dct_smape = dct_gen(trainAR, testAR, subm_24hrs, params=params)
        elif method == 'extra':
            fcst, extra_smape = extra_gen(trainAR, testAR, subm_24hrs, params=params)
        preds.append(np.ravel(fcst))

    # weighted sum
    y_pred = preds[0] * weight[0] + preds[1] * weight[1] + preds[2] * weight[2]
    submission.loc[key, 0:24] = np.ravel(y_pred)

# submission.to_csv('submit/submit_11.csv',index=True)
# val_smape.to_csv('val_results/val_smape_v2_with_tune.csv')