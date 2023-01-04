#
import os
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

dt = "/Users/ivan/Desktop/ALL/Data/TSolarStorm2"

N = 20
sample = 5000
fill = -999999


ddata = {}
with open(f"{dt}/train_output.txt", "r") as f:
    for i in tqdm(f):
        i = i.lstrip(" ").rstrip("\n").split(" ")
        ddata[i[0]] = i[2]

para = {}
with open(f"{dt}/train_para_input.txt", "r") as f:
    for i in tqdm(f):
        i = [_ for _ in i.lstrip(" ").rstrip("\n").split(" ") if _]
        para[i[0]] = [float(_) for _ in i[1:]]
with open(f"{dt}/test_para_input.txt", "r") as f:
    for i in tqdm(f):
        i = [_ for _ in i.lstrip(" ").rstrip("\n").split(" ") if _]
        para[i[0]] = [float(_) for _ in i[1:]]

_T = pd.DataFrame(para).T
print(_T)
print(_T.describe())

l = []
for k, v in para.items():
    l.append(len(v))
print(pd.value_counts(l))

raise Expression()


tdata, ttype = [], "train_jpg_input"
ipath = f"{dt}/{ttype}"
for ifits in tqdm(sorted(os.listdir(ipath))[:sample]):
    if ".jpg" in ifits:
        data = cv.imread(f"{ipath}/{ifits}")       
        tdata.append([fill if pd.isna(_) else _ for _ in np.resize(data, (N))] + para[ifits] + [int(ddata[ifits])])
tdata = np.array(tdata)

m_x = tdata[:,:-1]
m_y = tdata[:, -1]
print(pd.value_counts(m_y))

pca = decomposition.PCA(n_components=min(256, N//2))
m_x = pca.fit_transform(m_x)
print(m_x.shape)

X_trai, X_test, y_trai, y_test = train_test_split(m_x, m_y, test_size=0.50)
print(X_trai.shape, X_test.shape, y_trai.shape, y_test.shape)

d_train = xgb.DMatrix(X_trai, label=y_trai)
d_test = xgb.DMatrix(X_test, label=y_test)

params = {
    'booster': 'gblinear',
    'eval_metric': 'auc',
    'max_depth': 6,
    # 'subsample': 0.75,
    # 'colsample_bytree': 0.75,
    # 'min_child_weight': 2,
    # 'eta': 0.1,
    # 'silent': 1,
    'learning_rate': 0.10,
}

watchlist = [(d_train,'train'), (d_test, "test")]
bst = xgb.train(params, d_train, num_boost_round=100, evals=watchlist)
bst.save_model(f"../outs/m001.model")

y_trai_p = [1 if i > 0.5 else 0 for i in bst.predict(xgb.DMatrix(X_trai))]
y_test_p = [1 if i > 0.5 else 0 for i in bst.predict(xgb.DMatrix(X_test))]
print(pd.value_counts(y_trai), "\n", pd.value_counts(y_test))

a_trai = metrics.accuracy_score(y_trai, y_trai_p)
f_trai = metrics.f1_score(y_trai, y_trai_p, average='macro')
a_test = metrics.accuracy_score(y_test, y_test_p)
f_test = metrics.f1_score(y_test, y_test_p, average='macro')
print(">>> AC Train:%.6f, Tests:%.6f" % (a_trai, a_test))
print(">>> F1 Train:%.6f, Tests:%.6f" % (f_trai, f_test))
raise Expression()


idata, tdata, ttype = [], [], "test_jpg_input"
ipath = f"{dt}/{ttype}"
for ifits in tqdm(sorted(os.listdir(ipath))):
    if ".jpg" in ifits:
        data = cv.imread(f"{ipath}/{ifits}")  
        idata.append(ifits)
        tdata.append([fill if pd.isna(_) else _ for _ in np.resize(data, (N))] + para[ifits])
tdata = np.array(tdata)
# para[ifits]
rdata = [1 if i > 0.5 else 0 for i in bst.predict(xgb.DMatrix(pca.transform(tdata)))]
result = [int(i) for i in rdata]


with open(f"{dt}/outs/F_paipai.txt", "w") as f:
    for y1, y2 in zip(idata, result):
        f.write(f"{y1} {y2}\n")
#
# >>> AC Train:0.952635, Tests:0.951898
# >>> F1 Train:0.492076, Tests:0.490446



