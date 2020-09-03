#
import os
import tqdm
import numpy as np
import pandas as pd
c
from sklearn.model_selection import train_test_split

pdata0 = "outs/score/csv"

datal = []
for idata in [_ for _ in os.listdir(pdata0) if _.endswith(".csv")]:
    _data = pd.read_csv(f"{pdata0}/{idata}")
    datal.append(_data)

data = pd.concat(datal)
print(">>>", data.shape)

data.drop_duplicates(subset=["ipoit", "ideep", "idata"], inplace=True)
print(">>>", data.shape)

data["k"] = [f"{_1}_{_2}" for _1, _2 in zip(data["ipoit"], data["ideep"])]
score_col = [_ for _ in data.columns if _.startswith("m")] + ["bb_score1", "bb_score2"]
#for icol in score_col:
#    data[f"_{icol}"] = data[icol] * data["connected_domin_score"]
aggl = {f"{icol}": "sum" for icol in score_col}
aggl["Ascore"] = "max"
aggl["connected_domin_score"] = "max"
group = data.groupby(["k"]).agg(aggl)
group.reset_index(inplace=True)
group["Y"] = (group["Ascore"] - group["bb_score1"]*group["connected_domin_score"] - group["bb_score2"]*group["connected_domin_score"])/group["connected_domin_score"]
group.to_csv("tml_score_linear0.csv")

X = group[[f"{_}" for _ in data.columns if _.startswith("m")]]
Y = group["Y"]
print(X, Y, X.columns)

alphas_to_test = np.linspace(0.001, 1, 50)
lrModel = linear_model.RidgeCV(alphas=alphas_to_test, store_cv_values=True)
X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=0.5)
print(X1.shape, Y1.shape, X2.shape, Y2.shape)
lrModel.fit(X1, Y1)

alpha = lrModel.intercept_
beta = lrModel.coef_
print(alpha, "\n", beta.reshape((7, 5)), lrModel.alpha_)

y2y1 = [y2-y1 for y1, y2 in zip(lrModel.predict(X2), Y2)]
print("Dev:y2-y1 Max:%.6f, Min:%.6f, Avg:%.6f" % (np.max(y2y1), np.min(y2y1), np.mean(y2y1)))
print(lrModel.score(X1, Y1), lrModel.score(X2, Y2))
print({_i:_j for _i, _j in zip(X.columns, beta)})


result = {}
print("\n")
for icol in X.columns:
    imodel = linear_model.LinearRegression()
    imodel.fit(X1[[icol]], Y1)
    _1, _21, _22, _3, _4 = imodel.intercept_, imodel.coef_[0], imodel.coef_[0], imodel.score(X1[[icol]], Y1), imodel.score(X2[[icol]], Y2)
    result[f"{icol}_{icol}"] = [_1, _21, _22, _3, _4]
    print("%8s"%icol, "%8s"%icol, "%.12f"%_1, "%.12f"%_21, "%.12f"%_22, "%.6f"%_3, "%.6f"%_4)
    
print("\n")
for icol in X.columns:
    for jcol in X.columns:
        if icol != jcol:
            imodel = linear_model.LinearRegression()
            imodel.fit(X1[[icol, jcol]], Y1)
            _1, _21, _22, _3, _4 = imodel.intercept_, imodel.coef_[0], imodel.coef_[1], imodel.score(X1[[icol, jcol]], Y1), imodel.score(X2[[icol, jcol]], Y2)
            result[f"{icol}_{jcol}"] = [_1, _21, _22, _3, _4]
            print("%8s"%icol, "%8s"%icol, "%.12f"%_1, "%.12f"%_21, "%.12f"%_22, "%.6f"%_3, "%.6f"%_4)
_r = pd.DataFrame(result).T
_r.to_csv("tml_score_linear1.csv")

result = {}
print("\n")
for icol in X.columns:
    for jcol in X.columns:
        if icol != jcol:
            group["_"] = group[icol] + group[jcol]
            group["_"] = group["Y"]/group["_"]
            result[f"{icol}_{jcol}"] = [np.std(group["_"])]
            print("%8s"%icol, "%8s"%jcol, np.sum(group["_"]), np.mean(group["_"]))
_r = pd.DataFrame(result).T
_r.to_csv("tml_score_linear2.csv")



