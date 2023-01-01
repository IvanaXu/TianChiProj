
# -*-coding:utf-8-*-
# @auth ivan
# @time 2020-03-17 04:15:43
# @goal basic_model

import time
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from tqdm import tnrange
import pandas as pd
import numpy as np
import xgboost as xgb
import h5py


t0 = time.time()
d01t = pd.read_hdf(f"../user_data/tmp_data/data_t01.h5", key="data")
d01t = pd.DataFrame(d01t)

d01v = pd.read_hdf(f"../user_data/tmp_data/d_out_201807.h5", key="data")
d01vl = pd.DataFrame(pd.value_counts(d01v["mk"]).index)

# S 0.20
d01v1 = d01v[d01v["mk"].isin(d01vl.sample(frac=0.20)[0].values)]
# B 0.80
d01v2 = d01v[d01v["mk"].isin(d01vl.sample(frac=0.80)[0].values)]


t_num = pd.value_counts(d01t["bad"])
t_num0, t_num1 = t_num[0], t_num[1]
ns = t_num0/t_num1
var_l = d01t.columns.drop(["mk", "bad"])

data_x = d01t[var_l]
data_y = d01t["bad"]
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.5)

x_check1 = d01v1[var_l]
y_check1 = d01v1["bad"]

x_check2 = d01v2[var_l]
y_check2 = d01v2["bad"]

#
d_train = xgb.DMatrix(x_train, label=y_train)
d_test = xgb.DMatrix(x_test, label=y_test)
# params = {
#     'booster': 'gbtree',
#     'objective': 'binary:logistic',
#     'eval_metric': 'auc',
#     'max_depth': 4,
#     'lambda': 10,
#     'subsample': 0.75,
#     'colsample_bytree': 0.75,
#     'min_child_weight': 2,
#     'eta': 0.012,
#     'seed': 0,
#     'nthread': 8,
#     'silent': 1
# }

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 4,
    'lambda': 10,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_weight': 2,
    'eta': 0.0005,
    'seed': 0,
    'nthread': 10,
    'silent': 1,
    'lambda_l1': 0.00001,
    'lambda_l2': 0.00001,
    
    'tree_method': 'gpu_hist',
    'gpu_id': 1,
}
watchlist = [(d_train, 'train'), (d_test, "test")]
# 800, 0.68
bst = xgb.train(params, d_train, num_boost_round=25000, evals=watchlist)

i_max, n_max = 0, 0
x_train_y = bst.predict(xgb.DMatrix(x_train))
x_test_y = bst.predict(xgb.DMatrix(x_test))

for Ni in tnrange(1, 50, 2):
    N = Ni/100

    x_train_yp = [1 if i >= N else 0 for i in x_train_y]
    x_test_yp = [1 if i >= N else 0 for i in x_test_y]

    f11, f12 = metrics.f1_score(y_train, x_train_yp), metrics.f1_score(y_test, x_test_yp)
    print(
        "%f,%f,%f,%f,%f,%f,%f" % (N, (f11+f12)/2, f11, f12, 1/(ns+1), np.mean(x_train_yp), np.mean(x_test_yp)))
    i_loss = (f11 + f12) / 2
    if i_loss > i_max:
        i_max = i_loss
        n_max = N

N = n_max
print(">> A,Cut:%.6f, F1:%.8f" % (n_max, i_max*100))

Nr = N
# check1
x_check1_y = bst.predict(xgb.DMatrix(x_check1))
x_check1_yp = [1 if i >= Nr else 0 for i in x_check1_y]
check1_f1 = metrics.f1_score(y_check1, x_check1_yp)
print(pd.value_counts(d01v1["bad"]))
print(">> S,Cut:%.6f, F1:%.8f" % (Nr, check1_f1 * 100))

# check2
x_check2_y = bst.predict(xgb.DMatrix(x_check2))
x_check2_yp = [1 if i >= Nr else 0 for i in x_check2_y]
check2_f1 = metrics.f1_score(y_check2, x_check2_yp)
print(pd.value_counts(d01v2["bad"]))
print(">> B,Cut:%.6f, F1:%.8f" % (Nr, check2_f1 * 100))

# SAVE
with h5py.File('../user_data/tmp_data/value.h5', "w") as f:
    f["Nr"] = [Nr]
bst.save_model("../user_data/model_data/m002.model")

print("Use: %.6f min" % ((time.time()-t0)/60))

# 'eta': 0.012, num_boost_round = 800
# >> A,Cut:0.030000, F1:13.47124096
# >> S,Cut:0.030000, F1:4.85436893
# >> B,Cut:0.030000, F1:3.41555977



