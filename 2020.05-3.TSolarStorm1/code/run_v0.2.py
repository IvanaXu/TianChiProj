# -*-coding:utf-8-*-
# @auth ivan
# @time 2020-07-26 
# @goal Test

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn import metrics
from astropy.io import fits
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

dt = "/Users/ivan/Desktop/ALL/Data/TSolarStorm1"

# trainset, test_input
# continuum, magnetogram
# 标签1、2、3，分别表示磁类型alpha, beta, betax

dStype = {"alpha": 1, "beta": 2, "betax": 3}

N = 20
sample = None

ldata = []
ttype = "trainset"
for itype in ["continuum", "magnetogram"]:
    for idStype in ["alpha", "beta", "betax"]:
        ipath = f"{dt}/{ttype}/{itype}/{idStype}/"
        for ifits in os.listdir(ipath):
            ifitsn = ".".join(ifits.split(".")[:-2])
            ldata.append([ifitsn, itype, idStype])
ldata = sorted(ldata, key=lambda x: x[0])
print(ldata[:4])

tdata = {}
for ifitl in tqdm(ldata[:sample]):
    [ifitsn, itype, idStype] = ifitl
    data = fits.open(f"{dt}/{ttype}/{itype}/{idStype}/{ifitsn}.{itype}.fits")
    data.verify('fix')
    img_data = np.flipud(data[1].data)
    img_data = list(np.resize(img_data, (N)))

    if ifitsn not in tdata:
        tdata[ifitsn] = {itype: img_data+[dStype[idStype]]}
    else:
        tdata[ifitsn][itype] = img_data+[dStype[idStype]]

data = []
for k, v in tqdm(tdata.items()):
    if "continuum" in v and "magnetogram" in v:
        data.append([k]+v["continuum"]+v["magnetogram"])
coln = ["ifitsn"]+[f"C{_}" for _ in range(N)]+["T1"]+[f"M{_}" for _ in range(N)]+["T2"]
data = pd.DataFrame(data, columns=coln)
data["T"] = [i1==i2 for i1, i2 in zip(data["T1"], data["T2"])]
data["Target"] = [max(i1,i2) for i1, i2 in zip(data["T1"], data["T2"])]
print(pd.crosstab(data["T"], data["Target"]))

m_x = data[[f"C{_}" for _ in range(N)]+[f"M{_}" for _ in range(N)]]
m_x.fillna(-999999, inplace=True)
m_y = data["Target"]
print(m_x.shape, m_y.shape)

pca = decomposition.PCA(n_components=min(512, m_x.shape[1]))
m_x = pca.fit_transform(m_x)
print(m_x.shape)

X_trai, X_test, y_trai, y_test = train_test_split(m_x, m_y, test_size=0.20)
print(X_trai.shape, X_test.shape, y_trai.shape, y_test.shape)

d_train = xgb.DMatrix(X_trai, label=y_trai)
d_test = xgb.DMatrix(X_test, label=y_test)

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': max(dStype.values())+1,
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_weight': 2,
    'eta': 0.1,
    'silent': 1,
    'learning_rate': 0.2,
}

watchlist = [(d_train,'train'), (d_test, "test")]
bst = xgb.train(params, d_train, num_boost_round=200, evals=watchlist)
bst.save_model(f"../outs/m{itype}.model")

y_trai_p = bst.predict(xgb.DMatrix(X_trai))
y_test_p = bst.predict(xgb.DMatrix(X_test))

a_trai = metrics.accuracy_score(y_trai, y_trai_p)
f_trai = metrics.f1_score(y_trai, y_trai_p, average='macro')
a_test = metrics.accuracy_score(y_test, y_test_p)
f_test = metrics.f1_score(y_test, y_test_p, average='macro')
print(">>> AC Train:%.6f, Tests:%.6f" % (a_trai, a_test))
print(">>> F1 Train:%.6f, Tests:%.6f" % (f_trai, f_test))



ldata = []
ttype = "test_input"
for itype in ["continuum", "magnetogram"]:
    for idStype in ["/"]:
        ipath = f"{dt}/{ttype}/{itype}/{idStype}/"
        for ifits in os.listdir(ipath):
            ifitsn = ".".join(ifits.split(".")[:-2])
            ldata.append([ifitsn, itype, idStype])
ldata = sorted(ldata, key=lambda x: x[0])
print(ldata[:4])

tdata = {}
for ifitl in tqdm(ldata[:sample]):
    [ifitsn, itype, idStype] = ifitl
    data = fits.open(f"{dt}/{ttype}/{itype}/{idStype}/{ifitsn}.{itype}.fits")
    data.verify('fix')
    img_data = np.flipud(data[1].data)
    img_data = list(np.resize(img_data, (N)))

    if ifitsn not in tdata:
        tdata[ifitsn] = {itype: img_data}
    else:
        tdata[ifitsn][itype] = img_data

data = []
for k, v in tqdm(tdata.items()):
    if "continuum" in v and "magnetogram" in v:
        data.append([k]+v["continuum"]+v["magnetogram"])
coln = ["ifitsn"]+[f"C{_}" for _ in range(N)]+[f"M{_}" for _ in range(N)]
data = pd.DataFrame(data, columns=coln)

rdata = bst.predict(xgb.DMatrix(pca.transform(data[[f"C{_}" for _ in range(N)]+[f"M{_}" for _ in range(N)]])))
data["rnames"] = [i.split("_")[-2] for i in data["ifitsn"]]
data["result"] = [int(i) for i in rdata]
data.sort_values(by="rnames", inplace=True)

with open(f"{dt}/outs/C_paipai.txt", "w") as f:
    for y1, y2 in zip(data["rnames"], data["result"]):
        f.write(f"{y1} {y2}\n")



