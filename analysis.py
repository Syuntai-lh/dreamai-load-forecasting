import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# font
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = 'NanumGothic'
plt.rcParams['xtick.labelsize'] = 12.
plt.rcParams['ytick.labelsize'] = 12.

#%%
def preprocessing(df):
    df.index = df['Time']
    df.drop(columns=['Time'],inplace=True)
    df[df == 0] = np.nan
    return df

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train = preprocessing(train)
test = preprocessing(test)

# 24시간 맞춤
train = train.iloc[13:,:]

#%%
df = test
null_tr = (~pd.isnull(df)).sum(axis=0)
plt.plot(np.sort(null_tr))
plt.xticks()
plt.title('각 집에 대해 값이 있는 포인트 수')
plt.xlabel('house')
plt.show()

null_tr = (~pd.isnull(df)).sum(axis=1)
plt.plot(null_tr)
plt.xticks(null_tr.index[::2000], rotation=30)
plt.title('시간 별 값이 있는 가구 수')
plt.xlabel('Time')
plt.show()

#%%
null_tr.index[(null_tr > 7000)]


#%% accumulation 비율
def find_avg(start_i, end_i, avg):
    start_h = np.array(start_i) % 24 - 1
    end_h = np.array(end_i) % 24
    if start_h < 0:
        start_h = 23
    if start_h <= end_h:
        if end_h == 23:
            return avg[start_h:]
        else:
            return avg[start_h:end_h + 1]
    else:
        return np.concatenate((avg[start_h:], avg[:end_h + 1]))

def find_nan(homes, i, nan_l, nan_h, drop):
    home = homes.iloc[:, i].reset_index(drop=True)
    # zero를 nan으로 바꿈
    zero_index = (home.values == 0)
    home.loc[zero_index] = np.nan

    # nan의 start와 end를 구함
    start_i, end_i = -1, -1
    nan_val = home.isna().values
    nan_indexes = []
    for index, _ in enumerate(nan_val[:-1]):
        if nan_val[index] == False and nan_val[index + 1] == True:
            start_i = index + 1
            continue
        if nan_val[index] == True and nan_val[index + 1] == False:
            end_i = index
            # 처음 시작한 경우
            if start_i == -1:
                nan_indexes.append((0, end_i))
            else:
                nan_indexes.append((start_i, end_i))
            start_i, end_i = -1, -1
            continue
        # 마지막인 경우
        if index == len(nan_val) - 2 and start_i != -1:
            nan_indexes.append((start_i, index + 1))

    # nan이 nan_l <= ~ <= nan_h 일 경우 불러옴
    home = home.values
    # length and peak intensity
    # avg and sigma
    avg = np.nanmean(np.reshape(home, (-1, 24)), axis=0)
    sig = np.nanstd(np.reshape(home, (-1, 24)), axis=0)

    l_p_b = []  # 직전 값과의 corr
    start_i = []
    end_i = []
    for i, nan_index in enumerate(nan_indexes):
        nan_length = nan_index[1] - nan_index[0] + 1
        if (nan_length > nan_l) * (nan_length < nan_h):  # 조건을 만족할때만
            if i == 0:
                continue
            # if nan_index[0] in cand_index:
            #    continue

            # 3 sigma를 넘어갈 때만
            time = (nan_index[0] - 1) % 24
            tmp = home[nan_index[0] - 1] < avg[time] - 3 * sig[time] or home[nan_index[0] - 1] > avg[time] + 3 * sig[
                time]
            if drop:
                if tmp:
                    l_p_b.append((nan_length, home[nan_index[0] - 1]))
                    # nan_index_selected.append(nan_index)
                    start_i.append(nan_index[0])
                    end_i.append(nan_index[1])
            else:
                l_p_b.append((nan_length, home[nan_index[0] - 1]))
                # nan_index_selected.append(nan_index)
                start_i.append(nan_index[0])
                end_i.append(nan_index[1])

    return start_i, end_i, l_p_b

def find_normalized_intensity(homes, index, drop):
    start_i, end_i, l_p_b = find_nan(homes, index, 0, 14, drop)
    # find nan index
    tmp = lambda x: x[0]
    tmp2 = lambda x: x[1]
    length = []
    intensity = []
    for i in l_p_b: length.append(tmp(i))
    for i in l_p_b: intensity.append(tmp2(i))

    # find average
    home = homes.iloc[:, index].reset_index(drop=True)
    home = np.reshape(home.values, (-1, 24))
    avg = np.nanmean(home, axis=0)
    # print(avg)
    avgs = []
    avgs_2 = []
    for i, start in enumerate(start_i):
        avgs.append(find_avg(start, end_i[i], avg).mean())
        avgs_2.append(avg.mean())

    intensity_n = np.array(intensity) / np.array(avgs)
    intensity_n_2 = np.array(intensity) / np.array(avgs_2)
    return length, intensity, intensity_n.tolist(), intensity_n_2.tolist()

length_all = []
intensity_all = []
intensity_norm_all = []
length_all_2 = []
corrs = []
for i in range(1):
    # index = np.random.randint(10, 100)
    index = 5
    tmp1, tmp3, _, _ = find_normalized_intensity(train, index, drop=False)
    length_all = length_all + tmp1
    intensity_all = intensity_all + tmp3


plt.plot(length_all, intensity_all, '.', markersize=12)
print('===Done===')
plt.xlabel('Length of nan')
plt.ylabel('Before NaN')
plt.show()

#%%
models = ['non','lin','Mac','sim','lgb','svr','cat', 'dct',' extra', 'lstm']
result = pd.read_csv('val_results/saint_result.csv')
tmp = np.argmin(result.iloc[:,1:11].values,axis=1)
result['min_smape'] = np.nanmin(result.iloc[:,1:11].values, axis=1)
result['selected_model'] = [models[t] for t in tmp]
result.to_csv('saint_result.csv', index=False)

#%%
result = pd.read_csv('val_results/baseline_result.csv')
result.index = result.iloc[:,0]
result = result.iloc[:,1:10]
result['selected'] = result.columns[np.argmin(result.values, axis=1)]
model_rec = result['selected']
model_rec.to_csv('model_recognition.csv',index=True)

#%%
import pandas as pd
import pickle

def load_obj(name):
    with open('tune_results/' + name + '.pkl', 'rb') as f:
        trials = pickle.load(f)
        trials = sorted(trials, key=lambda k: k['loss'])
        return trials

baseline_val_result = pd.read_csv('val_results/baseline_result.csv')

#%%
for i in range(100):
    a = load_obj(str(i)+'_dnn')
    b = load_obj(str(i) + '_dnn_2')
    print('Reference {:.2f}, {:.2f} and {:.2f}'.format(baseline_val_result['min_smape'][i], a[0]['loss'], b[0]['loss']))

#%%
import pandas as pd
import numpy as np
def smape(true, pred):
    v = 2 * abs(pred - true) / (abs(pred) + abs(true))
    output = np.mean(v) * 100
    return output

a = pd.read_csv('submit/submit_8.csv')
b = pd.read_csv('submit/submit_10.csv')
# b = pd.read_csv('submit/result_00.csv')

smape_val = []
for i in range(200):
    result_1 = smape(np.ravel(a.iloc[i,1:25].values), np.ravel(b.iloc[i,1:25].values))
    smape_val.append(result_1)
    print(result_1)
print('====')
print(np.mean(smape_val))


#%%
def save_obj(obj, name):
    with open('tune_results/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

for i in range(200):
    a_list = load_obj(str(i)+'_dnn_3')
    for j in range(len(a_list)):
        a_list[j]['method'] = 'dnn_3'
    save_obj(a_list, str(i)+'_dnn_3')

#%%
from hyperopt import fmin, hp, tpe, Trials, space_eval
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample

space = {
            'EPOCH':                    hp.quniform('EPOCH', 10, 200, 5),
            'h1':                       hp.quniform('h1', 24, 24*20, 24),
            'h2':                       hp.choice('h2',
                                                  [
                                                      hp.quniform('h2', 24, 24 * 20, 24),
                                                      0
                                                  ]),
            'h3':                       hp.choice('h3',
                                                [
                                                    hp.quniform('h3', 24, 24 * 20, 24),
                                                    0
                                                ]),
            'h4':                       hp.choice('h4',
                                                [
                                                    hp.quniform('h4', 24, 24 * 20, 24),
                                                    0
                                                ]),
            'lr':                       hp.uniform('lr', 0.0001, 0.1),
            }

param = ho_sample(space)