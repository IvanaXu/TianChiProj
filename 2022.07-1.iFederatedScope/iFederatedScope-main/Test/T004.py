# score:3.3415
import sys
if len(sys.argv) > 1:
    datp = "/root/proj/data/CIKM22Competition/"
else:
    datp = "/Users/ivan/Library/CloudStorage/OneDrive-个人/Code/CIKM2022/data/CIKM22Competition/"

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def get_data(cid, data_type):
    _data = torch.load(f"{datp}/{cid}/{data_type}.pt")
    _lx, _ly = _data[0].x.shape[1], _data[0].y.shape[1]
    _ledge_attr = _data[0].edge_attr.shape[1] if "edge_attr" in _data[0].keys else 0

    ydata, index_data = [], []
    x1data, x2data, x3data, x4data = [], [], [], []
    e1data, e2data, e3data, e4data = [], [], [], []
    for idata in _data:
        x1data.append(np.max(np.array(idata.x), axis=0))
        x2data.append(np.mean(np.array(idata.x), axis=0))
        x3data.append(np.min(np.array(idata.x), axis=0))
        x4data.append(np.std(np.array(idata.x), axis=0))

        if _ledge_attr > 0:
            e1data.append(np.max(np.array(idata.edge_attr), axis=0))
            e2data.append(np.mean(np.array(idata.edge_attr), axis=0))
            e3data.append(np.min(np.array(idata.edge_attr), axis=0))
            e4data.append(np.std(np.array(idata.edge_attr), axis=0))

        ydata.append(np.array(idata.y)[0])
        index_data.append(idata.data_index)

    _data = pd.DataFrame([])
    _data["y"] = ydata
    _data["data_index"] = index_data

    _xcols = []
    for i in range(_lx):
        _data[f"x_1_{i}"] = np.array(x1data)[:, i]
        _xcols.append(f"x_1_{i}")

        _data[f"x_2_{i}"] = np.array(x2data)[:, i]
        _xcols.append(f"x_2_{i}")

        _data[f"x_3_{i}"] = np.array(x3data)[:, i]
        _xcols.append(f"x_3_{i}")

        _data[f"x_4_{i}"] = np.array(x4data)[:, i]
        _xcols.append(f"x_4_{i}")

    for i in range(_ledge_attr):
        _data[f"edge_attr_1_{i}"] = np.array(e1data)[:, i]
        _xcols.append(f"edge_attr_1_{i}")

        _data[f"edge_attr_2_{i}"] = np.array(e2data)[:, i]
        _xcols.append(f"edge_attr_2_{i}")

        _data[f"edge_attr_3_{i}"] = np.array(e3data)[:, i]
        _xcols.append(f"edge_attr_3_{i}")

        _data[f"edge_attr_4_{i}"] = np.array(e4data)[:, i]
        _xcols.append(f"edge_attr_4_{i}")

    return _data, _xcols, _ly


def get_model1():
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(random_state=10086)


def get_score1(_yt, _yp):
    from sklearn.metrics import accuracy_score
    return 1 - accuracy_score(_yt, _yp)


def get_predict1(x, mL):
    return [int(_r) for _r in np.mean([
        i_model.predict(x) for i_model in mL
    ], axis=0)]


def get_model2():
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(random_state=10086)


def get_score2(_yt, _yp):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(_yt, _yp)


def get_predict2(x, mL):
    return [float(_r) for _r in np.mean([
        i_model.predict(x) for i_model in mL
    ], axis=0)]


# cid, task_type, metric, K, model, score, predict
ids = [
    [1, ["cls", "Error rate", 10, get_model1, get_score1, get_predict1]],
    [2, ["cls", "Error rate", 8, get_model1, get_score1, get_predict1]],
    # 3, no edge_attr
    [3, ["cls", "Error rate", 8, get_model1, get_score1, get_predict1]],
    [4, ["cls", "Error rate", 4, get_model1, get_score1, get_predict1]],
    [5, ["cls", "Error rate", 8, get_model1, get_score1, get_predict1]],
    [6, ["cls", "Error rate", 6, get_model1, get_score1, get_predict1]],

    # 7, no edge_attr
    [7, ["cls", "Error rate", 10, get_model1, get_score1, get_predict1]],
    [8, ["cls", "Error rate", 8, get_model1, get_score1, get_predict1]],

    # 10/13, more Y
    [9, ["reg", "MSE", 6, get_model2, get_score2, get_predict2]],
    [10, ["reg", "MSE", 8, get_model2, get_score2, get_predict2]],
    [11, ["reg", "MSE", 8, get_model2, get_score2, get_predict2]],
    [12, ["reg", "MSE", 6, get_model2, get_score2, get_predict2]],
    [13, ["reg", "MSE", 8, get_model2, get_score2, get_predict2]],
]


result = []
for [cid, paras] in tqdm(ids):
    print(f"\nID {cid}:")
    [task_type, metric, K, model, score, predict] = paras

    train_data, xcols, ly = get_data(cid, "train")
    valis_data, _1, _2 = get_data(cid, "val")
    tests_data, _3, _4 = get_data(cid, "test")

    i_result = pd.DataFrame([cid for i in tests_data["data_index"]], columns=["client_id"])
    i_result["sample_id"] = tests_data["data_index"]

    print("\n")
    for iy in range(ly):
        train_X, train_Y = train_data[xcols], train_data["y"].apply(lambda x: x[iy])
        print(pd.value_counts(train_Y), train_X.shape, "\n")
        valis_X, valis_Y = valis_data[xcols], valis_data["y"].apply(lambda x: x[iy])
        tests_X = tests_data[xcols]

        modelL = []
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=K, shuffle=True, random_state=930721)
        for k, (i_train, i_tests) in enumerate(kf.split(train_X)):
            train_dataX1 = train_X.loc[i_train]
            train_dataX2 = train_X.loc[i_tests]
            train_dataY1 = train_Y.loc[i_train]
            train_dataY2 = train_Y.loc[i_tests]

            i_model = model()
            i_model.fit(train_dataX1, train_dataY1)
            modelL.append(i_model)

        print(f""">>> {cid} Y{iy} Train {metric}: {score(train_Y, predict(train_X, modelL)):.6f}""")
        print(f""">>> {cid} Y{iy} Valis {metric}: {score(valis_Y, predict(valis_X, modelL)):.6f}""")

        i_result[f"Y{iy}"] = predict(tests_X, modelL)
    result.append(i_result)

result = pd.concat(result)
result.to_csv(f"{datp}/result0.csv", index=False, header=False)
print(result.head(), result.shape)

with open(f"{datp}/result1.csv", "w") as f1:
    with open(f"{datp}/result0.csv", "r") as f0:
        for i in f0:
            i = i.strip("\n")
            i = ",".join([j[0] if j in ["0.0", "1.0"] else j for j in i.split(",") if j])
            f1.write(f"{i}\n")

