__author__ = 'C. Suphavilai'

import pandas as pd
import numpy as np
import sys, os
import time
import pickle
import random
from scipy import stats

import argparse
def predict_CaDRRes(ss_df,  cl_features_df, Xtest, ss_df_test,
                                           X, WP, WQ, mu, b_p, b_q, err_list, epoch):

    #ss_df = ytrain
    #cl_features_df = Xtrain
    #ss_df_test = ytest
    #Xtest = Xtest

    #read data
    f = WP.shape[1]
    P_train = list(ss_df.index)  # cell-line
    ss_test_df = ss_df_test.T
    cl_features_df = cl_features_df
    cl_features_df.index = cl_features_df.index.astype(str)
    #cl_list = list(cl_features_df.index.astype(str))
    #ss_test_df = ss_test_df[ss_test_df.index.isin(cl_list)]
    #X_train = np.matrix(cl_features_df.loc[P_train])
    M = len(list(ss_df.columns))
    #Y_test = ss_df_test
    Y_test = np.matrix(np.identity(len(b_q)))

    X_test = Xtest
    WQ = WQ
    WP = WP
    Y_train = ss_df
    X_train = cl_features_df


    P_list = list(ss_test_df.index)
    n_test = len(P_list)
    b_p = b_p
    b_q = b_q
    mu = mu

    #print('b_q', b_q.shape)


    ##### Estimate b_p_test #####
    num_seen_cl = len(set(P_list).intersection(ss_test_df.index))
    if num_seen_cl == n_test:
        P_train = np.array(P_list)
        b_p_test = np.zeros(ss_test_df.shape[0])
        for u, cl in enumerate(ss_test_df.index):
            if cl in P_train:
                cl_train_idx = np.argwhere(P_train == cl)[0][0]
                b_p_test[u] = b_p[cl_train_idx]
            else:
                print('ERROR: Unseen cell line, have to estimate b_p')
                sys.exit(1)
    # if not all cell lines are seen, then estimate biases for every cell line
    else:
        print('Estimating biases for unseen samples')
        b_p_test = np.matrix(b_p) * X_test.T

    ##### Calculate prediction #####
    #b_p_test = b_p

    #print('Y_test',Y_test.shape)
    #print('WQ',WQ.shape)
    #print('X-test',X_test.shape)
    #print('WP',WP.shape)

    Q_mat_test = Y_test * WQ
    P_mat_test = X_test @ WP

    #print(Q_mat_test.shape)
    #print(P_mat_test.shape)
    #print(P_mat_test.T.shape)
    #print(b_q.shape)

    temp = mu + (Q_mat_test * P_mat_test.T).T
    #print(temp.shape)
    #print(b_p_test.shape)
    temp = temp + b_q
    #print(temp.shape)
    b_p_test = b_p_test[:, np.newaxis]
    pred_mat = (temp.T + b_p_test).T
    return pred_mat

    '''
    Q_mat_train = Y_train * WQ
    P_mat_train = X_train @ WP

    temp = mu + (Q_mat_train * P_mat_train.T).T
    temp = temp + b_q
    train_pred_mat = (temp.T + b_p).T


    pred = b_q.T + (X @ WP) @ WQ.T
    pred = pred * -1  # convert sensitivity score to IC50
    return pred

    '''







'''
##############
# Parameters #
##############


##### Read the model #####


P_list = mdict['P_list']
Q_list = mdict['Q_list']

f = WP.shape[1]
P_train = list(ss_df.index)  # cell-line

out_dir = args.out_dir

##### Read data #####

ss_test_df = pd.read_csv(args.ss_test_name, index_col=0)
cl_features_df = pd.read_csv(args.cl_feature_fname, index_col=0)
cl_features_df.index = cl_features_df.index.astype(str)
cl_list = list(cl_features_df.index.astype(str))
ss_test_df = ss_test_df[ss_test_df.index.isin(cl_list)]

##############
# Prediction #
##############

new_out_fname = os.path.join(out_dir, 'CaDRReS_pred.csv')
new_out_dict_fname = os.path.join(out_dir, 'CaDRReS_pred.pickle')

P_test = list(ss_test_df.index)
n_test = len(P_test)
m_test = len(b_q)

X_test = np.matrix(cl_features_df.loc[P_test, P_list])
Y_test = np.matrix(np.identity(m_test))

##### Estimate b_p_test #####
num_seen_cl = len(set(P_list).intersection(ss_test_df.index))
if num_seen_cl == n_test:
    P_train = np.array(P_list)
    b_p_test = np.zeros(ss_test_df.shape[0])
    for u, cl in enumerate(ss_test_df.index):
        if cl in P_train:
            cl_train_idx = np.argwhere(P_train == cl)[0][0]
            b_p_test[u] = b_p[cl_train_idx]
        else:
            print 'ERROR: Unseen cell line, have to estimate b_p'
            sys.exit(1)
# if not all cell lines are seen, then estimate biases for every cell line
else:
    print 'Estimating biases for unseen samples'
    b_p_test = np.matrix(b_p) * X_test.T

##### Calculate prediction #####
Q_mat_test = Y_test * WQ
P_mat_test = X_test * WP

temp = mu + (Q_mat_test * P_mat_test.T).T
temp = temp + b_q
pred_mat = (temp.T + b_p_test).T





out_dict = {}
out_dict['P'] = P_mat_test
out_dict['Q'] = Q_mat_test
out_dict['mu'] = mu
out_dict['b_p'] = b_p_test
out_dict['b_q'] = b_q
pickle.dump(out_dict, open(new_out_dict_fname, 'w'))

pred_df = pd.DataFrame(pred_mat, columns=Q_list, index=ss_test_df.index)
# convert sensitivity score to IC50
pred_df *= -1
pred_df.to_csv(new_out_fname)
print 'Saved to', new_out_fname

'''