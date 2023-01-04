#
import os
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

N = 20000
sample = None

result = {}
for itype in ["continuum", "magnetogram"]:

    tdata, ttype = [], "trainset"
    for idStype in tqdm(dStype.keys(), desc=itype):
        ipath = f"{dt}/{ttype}/{itype}/{idStype}/"
        for ifits in tqdm(os.listdir(ipath)[:sample], desc=idStype):
            data = fits.open(f"{ipath}/{ifits}")
            data.verify('fix')
            img_data = np.flipud(data[1].data)            
            tdata.append(list(np.resize(img_data, (N)))+[dStype[idStype]])
    tdata = pd.DataFrame(tdata, columns=[f"V{i}" for i in range(N)]+["target"])
    
    m_x = tdata[tdata.columns.drop("target")]
    m_x.fillna(-999999, inplace=True)
    m_y = tdata["target"]

    pca = decomposition.PCA(n_components=256)
    m_x = pca.fit_transform(m_x)
    print(m_x.shape)

    X_trai, X_test, y_trai, y_test = train_test_split(m_x, m_y, test_size=0.50)
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
        'learning_rate': 0.1,
    }

    watchlist = [(d_train,'train'), (d_test, "test")]
    bst = xgb.train(params, d_train, num_boost_round=600, evals=watchlist)
    bst.save_model(f"../outs/m{itype}.model")

    y_trai_p = bst.predict(xgb.DMatrix(X_trai))
    y_test_p = bst.predict(xgb.DMatrix(X_test))

    a_trai = metrics.accuracy_score(y_trai, y_trai_p)
    f_trai = metrics.f1_score(y_trai, y_trai_p, average='macro')
    a_test = metrics.accuracy_score(y_test, y_test_p)
    f_test = metrics.f1_score(y_test, y_test_p, average='macro')
    print(">>> AC Train:%.6f, Tests:%.6f" % (a_trai, a_test))
    print(">>> F1 Train:%.6f, Tests:%.6f" % (f_trai, f_test))


    tdata, ttype = [], "test_input"
    ipath = f"{dt}/{ttype}/{itype}"
    for ifits in tqdm(sorted(os.listdir(ipath))):
        data = fits.open(f"{ipath}/{ifits}")
        data.verify('fix')
        img_data = np.flipud(data[1].data)            
        tdata.append(list(np.resize(img_data, (N))))
    tdata = pd.DataFrame(tdata, columns=[f"V{i}" for i in range(N)])

    rdata = bst.predict(xgb.DMatrix(pca.transform(tdata)))
    result[itype] = [int(i) for i in rdata]


# beta 2类的F1 score > betax 3类的F1 score > alpha 1类的F1 score
n = 1

def f123(y1, y2):
    if y1 == y2:
        return y1
    else:
        if y1 == 2 or y2 == 2:
            return 2
        if y1 == 3 or y2 == 3:
            return 3
        if y1 == 1 or y2 == 1:
            return 1

with open(f"{dt}/outs/C_paipai.txt", "w") as f:
    for y1, y2 in zip(result["continuum"], result["magnetogram"]):
        # print(str(n).zfill(6), y1, y2, f123(y1, y2))
        f.write(f"{str(n).zfill(6)} {f123(y1, y2)}\n")
        n += 1



