import pandas as pd  # 데이터 전처리
import numpy as np  # 데이터 전처리
import os
from tqdm import tqdm
try:
    import cPickle as pickle
except BaseException:
    import pickle

test = pd.read_csv('data/test.csv')
test['Time'] = pd.to_datetime(test.Time)
test = test.set_index('Time')

#%% model ==== Day prediction
def interp(datas):
    for i, data in enumerate(datas):
        if data == 0.:
            if i < 2:
                datas[i] = datas[i - 1]
            else:
                datas[i] = 0.2 * datas[i - 2] + 0.8 * datas[i - 1]
    return datas

def smape(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def transform_10days(data):
    trainAR, testAR = [], []
    window = 10
    i = 0
    while True:
        if i+window*2 <= data.shape[0]:
            trainAR.append(data[i:i+window])
            testAR.append(data[i+window:i+window*2])
            i += 1
        else:
            break
    trainAR = np.array(trainAR)
    testAR = np.array(testAR)
    return trainAR, testAR

#%%
smape_val_comp_3 = []
test_preds_3 = []
for key in tqdm(test.columns):
    fcst_d = np.zeros([1, 10])

    # load data
    home = test[key].copy()
    home[pd.isnull(home)] = 0.
    home = home.values
    first_non_nan_idx = np.where(home != 0.)[0][0]
    home = home[first_non_nan_idx:]
    cut = home.shape[0] % 24
    home = home[cut:]

    home = interp(home)

    home = np.reshape(home, (-1, 24))
    home = home.sum(axis=1)

    trainAR, testAR = transform_10days(home)

    split = int(trainAR.shape[0] * 0.85)
    X_tr, y_tr = trainAR[:split, :], testAR[:split, :]
    X_val, y_val = trainAR[split:, :], testAR[split:, :]

    # similar day approach
    y_pred = np.zeros(y_val.shape)
    len_ = X_val.shape[0]
    for test_idx in range(len_):
        idxs = np.argsort(np.linalg.norm(X_tr - X_val[test_idx,:], axis=1))
        most_similar_day_idx = idxs[:2]
        y_pred[test_idx,:] = np.mean(X_tr[most_similar_day_idx,:], axis=0)

    smape_val = np.zeros((len_))
    for i in range(len_):
        smape_val[i] = smape(np.ravel(y_pred[i, :]), np.ravel(y_val[i, :]))

    smape_val_comp_3.append(np.mean(smape_val))
    test_preds_3.append(y_pred)