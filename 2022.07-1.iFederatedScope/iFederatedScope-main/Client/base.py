import os
import gzip
import pickle
from config import *
import numpy as np
import pandas as pd
from tqdm import tqdm

datp = "/Users/ivan/Desktop/Data/CIKM2022/CIKM22Competition/"
pklp = f"{datp}/X"
clientp = f"{datp}/Client"

# TEST = True
TEST = False

# # cid, task_type, metric, K, model, score, predict
if TEST:
    ids = [
        # [1, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        # [2, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        # [3, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        # [4, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        # [5, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        # [6, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        # [7, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        # [8, ["cls", "Error rate", k, get_model1, get_score1, get_predict1]] for k in range(2, 21)
        # [9, ["reg", "MSE", k, get_model2, get_score2, get_predict2]] for k in range(2, 21)
        # [10, ["reg", "MSE", k, get_model2, get_score2, get_predict2]] for k in range(2, 21)
        # [11, ["reg", "MSE", k, get_model2, get_score2, get_predict2]] for k in range(2, 21)
        # [12, ["reg", "MSE", k, get_model2, get_score2, get_predict2]] for k in range(2, 21)
        # [13, ["reg", "MSE", k, get_model2, get_score2, get_predict2]] for k in range(2, 21)
    ]
else:
    ids = [
        [1, ["cls", "Error rate", 13, get_model1, get_score1, get_predict1]],
        [2, ["cls", "Error rate", 8, get_model1, get_score1, get_predict1]],
        # 3, no edge_attr
        [3, ["cls", "Error rate", 3, get_model1, get_score1, get_predict1]],
        [4, ["cls", "Error rate", 5, get_model1, get_score1, get_predict1]],
        [5, ["cls", "Error rate", 11, get_model1, get_score1, get_predict1]],
        [6, ["cls", "Error rate", 2, get_model1, get_score1, get_predict1]],
        # 7, no edge_attr
        [7, ["cls", "Error rate", 2, get_model1, get_score1, get_predict1]],
        [8, ["cls", "Error rate", 11, get_model1, get_score1, get_predict1]],

        # 10/13, more Y
        [9, ["reg", "MSE", 3, get_model2, get_score2, get_predict2]],
        [10, ["reg", "MSE", 2, get_model2, get_score2, get_predict2]],
        [11, ["reg", "MSE", 3, get_model2, get_score2, get_predict2]],
        [12, ["reg", "MSE", 4, get_model2, get_score2, get_predict2]],
        [13, ["reg", "MSE", 2, get_model2, get_score2, get_predict2]],
    ]


min_train_valis = np.inf
result, record = [], []
for [cid, paras] in ids:
    print(f"\nID {cid}:")
    [task_type, metric, K, model, score, predict] = paras

    train_data = pd.read_pickle(f"{pklp}/X-{cid}-train.pkl", compression="zip")
    valis_data = pd.read_pickle(f"{pklp}/X-{cid}-val.pkl", compression="zip")
    tests_data = pd.read_pickle(f"{pklp}/X-{cid}-test.pkl", compression="zip")
    print(train_data.shape, valis_data.shape, tests_data.shape)

    xcols, ly = COLS, train_data["y_l"].max()
    i_result = pd.DataFrame([cid for i in tests_data["data_index"]], columns=["client_id"])
    i_result["sample_id"] = tests_data["data_index"]

    iy, train_scoreL, valis_scoreL, modelD = 0, [], [], {}
    for iy in tqdm(range(ly)):
        train_X, train_Y = train_data[xcols], train_data["y"].apply(lambda x: x[iy])
        # print(pd.value_counts(train_Y), train_X.shape, "\n")
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

            i_model = model(cid)
            i_model.fit(train_dataX1, train_dataY1)
            modelL.append(i_model)

        train_score, valis_score = score(train_Y, predict(train_X, modelL)), score(valis_Y, predict(valis_X, modelL))

        i_result[f"Y{iy}"] = predict(tests_X, modelL)
        train_scoreL.append(train_score)
        valis_scoreL.append(valis_score)
        modelD[f"Y{iy}"] = modelL

    if True:
        train_score, valis_score = np.mean(train_scoreL), np.mean(valis_scoreL)
        std_train_valis = np.std([train_score, valis_score])
        print(
            f""">>> {cid} Y-AVG /{K} {metric}"""
            f""" Train: {train_score:.6f}"""
            f""" Valis: {valis_score:.6f}"""
            f""" STD: {std_train_valis:.6f}"""
            f""" {"✅" if TEST and min_train_valis > std_train_valis else ""}"""
        )
        if min_train_valis > std_train_valis:
            min_train_valis = std_train_valis

    with gzip.GzipFile(f"{clientp}/base/{cid}-modelD.mdl", "wb") as f:
        pickle.dump(modelD, f)

    record.extend([train_score, valis_score])
    result.append(i_result)

with open(".record", "w") as f:
    for i in record:
        f.write(f"{i:.6f}\n")

result = pd.concat(result)
result.to_csv(f"{datp}/Result/result-b0.csv", index=False, header=False)
print(result.head(), result.shape)

with open(f"{datp}/Result/result-b1.csv", "w") as f1:
    with open(f"{datp}/Result/result-b0.csv", "r") as f0:
        for i in f0:
            i = i.strip("\n")
            i = ",".join([j[0] if j in ["0.0", "1.0"] else j for j in i.split(",") if j])
            f1.write(f"{i}\n")

os.system('say "i finish the job"')
