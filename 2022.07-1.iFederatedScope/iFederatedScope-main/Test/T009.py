import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

datp = "/Users/ivan/Desktop/Data/CIKM2022/CIKM22Competition/"

x = []
for cid in [8]:
    _data = torch.load(f"{datp}/{cid}/train.pt")

    for idata in tqdm(_data):
        jdata = str(list(np.array(idata.edge_index[1])))

        x.append([
            1 if f"{i1}, {i2}" in jdata else 0
            for i1 in range(120)
            for i2 in range(120)
        ] + [np.array(idata.y)[0][0]])

x = pd.DataFrame(x, columns=[f"{i1}-{i2}" for i1 in range(120) for i2 in range(120)]+["y"])
print(x)

corr = []
for icol in tqdm(x.columns):
    if icol != "y":
        corr.append([icol, abs(np.corrcoef(x[icol], x["y"])[0][1]), x[icol].mean()])
        # break

corr = pd.DataFrame(corr)
corr[3] = corr[1] + corr[2]
print(
    list(corr.sort_values([1], ascending=False).head(2)[0].values) +
    list(corr.sort_values([3], ascending=False).head(2)[0].values)
)

corr = corr.sort_values([1], ascending=False)
l_corr = list(corr.head(100)[0].values)
print(l_corr)


def get_model1():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective="regression",
        bagging_fraction=0.80,
        feature_fraction=0.80,
        max_depth=9,
        n_estimators=100,
        verbose=-1,
        n_jobs=-1,
        learning_rate=0.1,
    )


def get_score1(_yt, _yp):
    from sklearn.metrics import accuracy_score
    return 1 - accuracy_score(_yt, _yp)


i_model = get_model1()
train_X = x[l_corr]
train_Y = x["y"]
i_model.fit(train_X, train_Y)
print(get_score1(train_Y, i_model.predict(train_X)))
