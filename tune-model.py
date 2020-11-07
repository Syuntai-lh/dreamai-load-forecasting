"""
Created on Mon Mar  2 22:44:55 2020

@author: guseh
"""
# packages
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL
from functools import partial
import pandas as pd
# models
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from util_saint import *

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pickle
def save_obj(obj, name):
    with open('tune_results/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('tune_results/' + name + '.pkl', 'rb') as f:
        trials = pickle.load(f)
        trials = sorted(trials, key=lambda k: k['loss'])
        return trials

def make_param_int(param, key_names):
    for key, value in param.items():
        if key in key_names:
            param[key] = int(param[key])
    return param


class Tuning_model(object):

    def __init__(self):
        self.random_state = 0
        self.space = {}

    # parameter setting
    def rf_space(self):
        self.space =  {
            'max_depth':                hp.quniform('max_depth',1, 20,1),
            'min_samples_leaf':         hp.quniform('min_samples_leaf', 1,10,1),
            'min_samples_split':        hp.uniform('min_samples_split', 0,1),
            'n_estimators':             hp.quniform('n_estimators', 100,1000,1),
            'criterion':                hp.choice('criterion', ['mse', 'mae']),
            'random_state' :            self.random_state,
            'n_jobs': -1
           }

    def svr_space(self):
        self.space = {
            'kernel':                   hp.choice('kernel',['linear', 'poly', 'rbf', 'sigmoid']),
            'C':                        hp.uniform('C',1,10),
            'gamma':                    hp.loguniform('gamma',np.log(1e-7),np.log(1e-1)),
            'epsilon':                  hp.uniform('epsilon',0,1),
            }

    def dct_space(self):
        self.space = {
            'max_depth':                hp.quniform('max_depth', 2, 20, 1),
            'min_samples_leaf':         hp.quniform('min_samples_leaf', 1, 10, 1),
            'min_samples_split':        hp.uniform('min_samples_split', 0, 1),
            'criterion':                hp.choice('criterion', ['mse', 'mae']),
            }

    def extra_space(self):
        self.space = {
            'max_depth':                hp.quniform('max_depth', 2, 20, 1),
            'min_samples_leaf':         hp.quniform('min_samples_leaf', 1, 10, 1),
            'min_samples_split':        hp.uniform('min_samples_split', 0, 1),
            'criterion':                hp.choice('criterion', ['mse', 'mae']),
            'n_jobs':                   -1
            }

    def dnn_space(self):
        self.space = {
            'EPOCH':                    hp.quniform('EPOCH', 50, 100, 5),
            'h1':                       hp.quniform('h1', 24, 24*20, 24),
            'h2':                       hp.quniform('h2', 24, 24*20, 24),
            'h3':                       hp.quniform('h3', 24, 24*20, 24),
            'lr':                       hp.uniform('lr', 0.0001, 0.01),
            }

    # optimize
    def process(self, clf_name, train_set, trials, algo, max_evals):
        fn = getattr(self, clf_name+'_val')
        space = getattr(self, clf_name+'_space')
        space()
        fmin_objective = partial(fn, train_set=train_set)
        try:
            result = fmin(fn=fmin_objective, space=self.space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def rf_val(self, params, train_set):
        trainAR, testAR = train_set
        params = make_param_int(params,['max_depth','max_features','n_estimators','min_samples_leaf'])
        _, smape = machine_learn_gen(trainAR, testAR, subm_24hrs, params)
        # Dictionary with information for evaluation
        return {'loss': smape, 'params': params, 'status': STATUS_OK, 'method':args.method}

    def svr_val(self, params, train_set):
        trainAR, testAR = train_set
        _, smape = svr_gen(trainAR, testAR, subm_24hrs, params)
        # Dictionary with information for evaluation
        return {'loss': smape, 'params': params, 'status': STATUS_OK, 'method':args.method}

    def dct_val(self, params, train_set):
        params = make_param_int(params, ['max_depth', 'min_samples_leaf'])
        trainAR, testAR = train_set
        _, smape = dct_gen(trainAR, testAR, subm_24hrs, params)
        # Dictionary with information for evaluation
        return {'loss': smape, 'params': params, 'status': STATUS_OK, 'method':args.method}

    def extra_val(self, params, train_set):
        params = make_param_int(params, ['max_depth', 'min_samples_leaf'])
        trainAR, testAR = train_set
        _, smape = extra_gen(trainAR, testAR, subm_24hrs, params)
        # Dictionary with information for evaluation
        return {'loss': smape, 'params': params, 'status': STATUS_OK, 'method':args.method}

    def dnn_val(self, params, train_set):
        params = make_param_int(params, ['EPOCH', 'h1', 'h2', 'h3'])
        trainAR, testAR = train_set
        _, smape = non_linear_model_gen(trainAR, testAR, params)
        # Dictionary with information for evaluation
        return {'loss': smape, 'params': params, 'status': STATUS_OK, 'method':args.method}


if __name__ == '__main__':

    # load config
    parser = argparse.ArgumentParser(description='Tune each household...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', default='dnn', choices=['rf', 'svr', 'dct','extra','dnn'])
    parser.add_argument('--max_evals', default=1000, type=int)
    parser.add_argument('--save_file', default='tmp')
    parser.add_argument('--col_idx', default=0, type=int)
    args = parser.parse_args()

    # load dataset
    test = pd.read_csv('data/test.csv')
    submission = pd.read_csv('submit/submission.csv')
    test['Time'] = pd.to_datetime(test.Time)
    test = test.set_index('Time')
    key = test.columns[args.col_idx]
    with open('data_pr/' + key + '.pkl', 'rb') as f:
        data_pr = pickle.load(f)
    train, train_label, subm_24hrs, fchk = data_pr

    # main
    clf = args.method
    bayes_trials = Trials()
    obj = Tuning_model()
    tuning_algo = tpe.suggest # -- bayesian opt
    # tuning_algo = tpe.rand.suggest # -- random search
    obj.process(args.method, [train, train_label],
                           bayes_trials, tuning_algo, args.max_evals)

    # save trial
    save_obj(bayes_trials.results,args.save_file)