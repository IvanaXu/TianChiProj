import gzip
import pickle
import torch
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from gensim.models import Word2Vec
mdlf = "/Users/ivan/Desktop/Data/CIKM2022/CIKM22Competition/Server/NLP.model"
NLP = Word2Vec.load(mdlf)
NLP_c = 256

YN = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1,
    6: 1, 7: 1, 8: 1, 9: 1, 10: 10,
    11: 1, 12: 1, 13: 12,
}
COLS = ["e_x", "e_l", "y_l"] + \
       [f"x_{n}_{i}" for i in range(38) for n in range(7)] + \
       [f"e_{n}_{i}" for i in range(8) for n in range(7)] + \
       [f"ei_{n}_{i}" for i in range(111+1) for n in range(3)] + \
       [f"cid_y_{i1}-{j}" for i1, i2 in YN.items() for j in range(i2)] + \
       [f"em_{i}" for i in range(NLP_c)]


def get_data(path1, path2, cid, data_type, top=0, is_NLP=""):
    _data = torch.load(f"{path1}/{cid}/{data_type}.pt")
    _lx, _ly = _data[0].x.shape[1], _data[0].y.shape[1]
    _ledge_attr = _data[0].edge_attr.shape[1] if "edge_attr" in _data[0].keys else 0

    #
    ydata, index_data, lxdata, ledata = [], [], [], []
    ei0data, ei1data, ei2data = [], [], []
    ej0data = []
    xdatal = [x1data, x2data, x3data, x4data, x5data, x6data, x7data] = [[] for _ in range(7)]
    edatal = [e1data, e2data, e3data, e4data, e5data, e6data, e7data] = [[] for _ in range(7)]

    for idata in _data:
        x1data.append(np.max(np.array(idata.x), axis=0))
        x2data.append(np.mean(np.array(idata.x), axis=0))
        x3data.append(np.min(np.array(idata.x), axis=0))
        x4data.append(np.std(np.array(idata.x), axis=0))
        x5data.append(np.percentile(np.array(idata.x), q=25, axis=0))
        x6data.append(np.percentile(np.array(idata.x), q=50, axis=0))
        x7data.append(np.percentile(np.array(idata.x), q=75, axis=0))

        if _ledge_attr > 0:
            e1data.append(np.max(np.array(idata.edge_attr), axis=0))
            e2data.append(np.mean(np.array(idata.edge_attr), axis=0))
            e3data.append(np.min(np.array(idata.edge_attr), axis=0))
            e4data.append(np.std(np.array(idata.edge_attr), axis=0))
            e5data.append(np.percentile(np.array(idata.edge_attr), q=25, axis=0))
            e6data.append(np.percentile(np.array(idata.edge_attr), q=50, axis=0))
            e7data.append(np.percentile(np.array(idata.edge_attr), q=75, axis=0))

        ei0 = list(np.array(idata.edge_index)[0])
        ei0data.append([1 if i0 in ei0 else 0 for i0 in range(top+1)])
        ei1 = list(np.array(idata.edge_index)[1])
        ei1data.append([1 if i1 in ei1 else 0 for i1 in range(top+1)])
        ei2 = list(np.array(idata.edge_index)[0]) + list(np.array(idata.edge_index)[1])
        ei2data.append([1 if i2 in ei2 else 0 for i2 in range(top+1)])

        ej0 = [str(i) for i in np.array(idata.edge_index)[1]]
        ej0data.append(np.mean(NLP.wv[ej0], axis=0))

        ydata.append(np.array(idata.y)[0])
        index_data.append(idata.data_index)
        lxdata.append(idata.x.shape[0])
        ledata.append(idata.edge_index.shape[1])

    _data = pd.DataFrame([])
    _data["y"] = ydata
    _data["data_index"] = index_data

    _data["e_x"] = lxdata
    _data["e_l"] = ledata
    _data["y_l"] = [len(i) for i in ydata]

    _xcols = ["e_x", "e_l", "y_l"]
    for i in range(_lx):
        for n, xdata in enumerate(xdatal):
            _data[f"x_{n}_{i}"] = np.array(xdata)[:, i]
            _xcols.append(f"x_{n}_{i}")

    for i in range(_ledge_attr):
        for n, edata in enumerate(edatal):
            _data[f"e_{n}_{i}"] = np.array(edata)[:, i]
            _xcols.append(f"e_{n}_{i}")

    for i in range(top+1):
        for n, eidata in enumerate([ei0data, ei1data, ei2data]):
            _data[f"ei_{n}_{i}"] = np.array(eidata)[:, i]
            _xcols.append(f"ei_{n}_{i}")

    for i1, i2 in YN.items():
        for j in range(i2):
            _data[f"cid_y_{i1}-{j}"] = -1
            _xcols.append(f"cid_y_{i1}-{j}")

    for i in range(NLP_c):
        _data[f"em_{i}"] = np.array(ej0data)[:, i] if is_NLP else -1
        _xcols.append(f"em_{i}")

    # COLS
    assert len(set(_xcols) - set(COLS)) == 0
    for icol in COLS:
        if icol not in _xcols:
            _data[icol] = -1

    # to_pkl
    _data.to_pickle(f"{path2}/X-{cid}-{data_type}.pkl", compression="zip")
    return _data


def get_model1(cid):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective="regression",
        bagging_fraction=0.50 if cid in [2, 4] else 0.90,
        feature_fraction=0.50 if cid in [2, 4] else 0.90,
        max_depth=10,
        n_estimators=100,
        verbose=-1,
        n_jobs=-1,
    )


def get_score1(_yt, _yp):
    from sklearn.metrics import accuracy_score
    return 1 - accuracy_score(_yt, _yp)


def get_predict1(x, mL):
    return [int(_r) for _r in np.mean([
        i_model.predict(x) for i_model in mL
    ], axis=0)]


def get_model2(cid):
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        objective="regression",
        bagging_fraction=0.50 if cid in [2, 4] else 0.90,
        feature_fraction=0.50 if cid in [2, 4] else 0.90,
        max_depth=10,
        n_estimators=100,
        verbose=-1,
        n_jobs=-1,
    )


def get_score2(_yt, _yp):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(_yt, _yp)


def get_predict2(x, mL):
    return [float(_r) for _r in np.mean([
        i_model.predict(x) for i_model in mL
    ], axis=0)]


def update(df, dfcid, mdlp):
    for i1, i2 in YN.items():
        if i1 == dfcid:
            continue

        with gzip.GzipFile(f"{mdlp}/{i1}-modelD.mdl", "rb") as f:
            modelD = pickle.load(f)

        predict = get_predict2
        for j in range(i2):
            df[f"cid_y_{i1}-{j}"] = predict(df[COLS], modelD[f"Y{j}"])
    return df
