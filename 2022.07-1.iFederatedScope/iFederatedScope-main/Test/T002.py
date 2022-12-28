# Test submission
import sys
if len(sys.argv) > 1:
    datp = "/root/proj/data/CIKM22Competition/"
else:
    datp = "/Users/ivan/Library/CloudStorage/OneDrive-个人/Code/CIKM2022/data/CIKM22Competition/"

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

ids = range(1, 13+1)
# ids = [1, 13]


def get_data(cid, data_type):
    _data = torch.load(f"{datp}/{cid}/{data_type}.pt")
    _lx, _ly = _data[0].x.shape[1], _data[0].y.shape[1]
    _xcols = [f"x{i}" for i in range(_lx)]

    xdata, ydata, index_data = [], [], []
    for idata in _data:
        xdata.append(np.mean(np.array(idata.x), axis=0))
        ydata.append(np.array(idata.y)[0])
        index_data.append(idata.data_index)

    _data = pd.DataFrame(xdata, columns=_xcols)
    _data["y"] = ydata
    _data["data_index"] = index_data
    return _data, _xcols, _ly


result = []
for cid in tqdm(ids):
    print(f"\nID {cid}:")
    train_data, xcols, ly = get_data(cid, "train")
    valis_data, _1, _2 = get_data(cid, "val")
    tests_data, _3, _4 = get_data(cid, "test")

    iresult = pd.DataFrame([cid for i in tests_data["data_index"]], columns=["client_id"])
    iresult["sample_id"] = tests_data["data_index"]

    print()
    if cid < 9:
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()

        train_X, train_Y = train_data[xcols], train_data["y"].apply(lambda x: x[0])
        valis_X, valis_Y = valis_data[xcols], valis_data["y"].apply(lambda x: x[0])
        tests_X = tests_data[xcols]
        model.fit(train_X, train_Y)

        from sklearn.metrics import accuracy_score
        score = 1 - accuracy_score(train_Y, model.predict(train_X))
        print(f""">>> {cid} Train Error rate: {score:.6f}""")

        score = 1 - accuracy_score(valis_Y, model.predict(valis_X))
        print(f""">>> {cid} Valis Error rate: {score:.6f}""")

        iresult["Y0"] = model.predict(tests_X)
    else:
        for iy in range(ly):
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor()

            train_X, train_Y = train_data[xcols], train_data["y"].apply(lambda x: x[iy])
            valis_X, valis_Y = valis_data[xcols], valis_data["y"].apply(lambda x: x[iy])
            tests_X = tests_data[xcols]
            model.fit(train_X, train_Y)

            from sklearn.metrics import mean_squared_error
            score = mean_squared_error(train_Y, model.predict(train_X))
            print(f""">>> {cid} Y{iy} MSE: {score:.6f}""")

            score = mean_squared_error(valis_Y, model.predict(valis_X))
            print(f""">>> {cid} Y{iy} MSE: {score:.6f}""")

            iresult[f"Y{iy}"] = model.predict(tests_X)
    result.append(iresult)

result = pd.concat(result)
result.to_csv(f"{datp}/result0.csv", index=False, header=False)
print(result)

with open(f"{datp}/result1.csv", "w") as f1:
    with open(f"{datp}/result0.csv", "r") as f0:
        for i in f0:
            i = i.strip("\n")
            i = ",".join([j[0] if j in ["0.0", "1.0"] else j for j in i.split(",") if j])
            f1.write(f"{i}\n")
