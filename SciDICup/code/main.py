#
import os
import random
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

dt = "../data/AstroSet"

# F:1, M:2
dv = {"flare star": 1, "microlensing": 2, "normal": 0}
dlist = pd.read_csv(
    f"{dt}/train_abnormal",
    sep=",",
    header=None,
    names=["starid", "type", "path"])

dlist = {k: dv[v] for k, v in zip(dlist["starid"], dlist["type"])}
print(dlist)

klist = {}

for i in dlist.keys():
    title = i[4:22]
    if title not in klist:
        klist[title] = [i]
    else:
        klist[title].append(i)
print(klist)

sample = 2000 # TODO:


def JD2datetime(jd):
    start_day = datetime.datetime(1858, 11, 17, 0, 0, 0)
    std_dt = start_day + datetime.timedelta(days=jd - 2400000.5)
    return std_dt.timestamp()


JD2datetime(2458490.2478008997)

n = 0
alldata = []

for isky in os.listdir(f"{dt}"):
    if isky.startswith("0"):
        n += 1

        data = pd.read_csv(
            f"{dt}/{isky}/abstract",
            sep=" ",
            header=None,
            names=["starid", "ra", "dec", "length"])

        rstar = os.listdir(f"{dt}/{isky}")
        random.shuffle(rstar)
        jstar = klist[isky] if isky in klist else []

        for istar in tqdm(rstar[:sample] + jstar, desc=f"{n}_{isky}"):
            if istar.startswith("ref"):
                try:
                    _idata = [istar]

                    idata = pd.read_csv(
                        f"{dt}/{isky}/{istar}",
                        sep=" ",
                        header=None,
                        names=["jd", "magnorm", "mage"])

                    idata["jd_c"] = idata["jd"].apply(JD2datetime)
                    idata["magnorm_m0"] = [
                        i - j for i, j in zip(idata["magnorm"], idata["mage"])
                    ]
                    idata["magnorm_m1"] = [
                        i + j for i, j in zip(idata["magnorm"], idata["mage"])
                    ]

                    mean_magnorm = np.mean(idata["magnorm"])
                    mean_magnorm_m0 = np.mean(idata["magnorm_m0"])
                    mean_magnorm_m1 = np.mean(idata["magnorm_m1"])
                    idata["magnorm_dmean"] = [
                        1 if i - mean_magnorm > 0 else 0
                        for i in idata["magnorm"]
                    ]
                    idata["magnorm_m0_dmean"] = [
                        1 if i - mean_magnorm_m0 > 0 else 0
                        for i in idata["magnorm_m0"]
                    ]
                    idata["magnorm_m1_dmean"] = [
                        1 if i - mean_magnorm_m1 > 0 else 0
                        for i in idata["magnorm_m1"]
                    ]

                    for col in [
                            "jd", "jd_c", "magnorm", "mage", "magnorm_m0",
                            "magnorm_m1", "magnorm_dmean", "magnorm_m0_dmean",
                            "magnorm_m1_dmean"
                    ]:
                        for nn in range(1, 60):
                            idata[f"{col}_{nn}_d1"] = [
                                i - j for i, j in zip(idata[col], idata[col].
                                                      shift(nn))
                            ]
                            idata[f"{col}_{nn}_d2"] = [
                                1 if (i - j) > 0 else 0 for i, j in zip(
                                    idata[col], idata[col].shift(nn))
                            ]

                            _idata.append(np.sum(idata[f"{col}_{nn}_d1"]))
                            _idata.append(np.max(idata[f"{col}_{nn}_d1"]))
                            _idata.append(np.mean(idata[f"{col}_{nn}_d1"]))
                            _idata.append(np.min(idata[f"{col}_{nn}_d1"]))
                            _idata.append(np.median(idata[f"{col}_{nn}_d1"]))
                            _idata.append(
                                np.max(idata[f"{col}_{nn}_d1"]) -
                                np.min(idata[f"{col}_{nn}_d1"]))
                            _idata.append(
                                np.max(idata[f"{col}_{nn}_d1"]) -
                                np.mean(idata[f"{col}_{nn}_d1"]))
                            _idata.append(
                                np.max(idata[f"{col}_{nn}_d1"]) -
                                np.median(idata[f"{col}_{nn}_d1"]))
                            _idata.append(
                                np.mean(idata[f"{col}_{nn}_d1"]) -
                                np.median(idata[f"{col}_{nn}_d1"]))
                            _idata.append(np.std(idata[f"{col}_{nn}_d1"]))

                            _idata.append(np.sum(idata[f"{col}_{nn}_d2"]))
                            _idata.append(np.max(idata[f"{col}_{nn}_d2"]))
                            _idata.append(np.mean(idata[f"{col}_{nn}_d2"]))
                            _idata.append(np.min(idata[f"{col}_{nn}_d2"]))
                            _idata.append(np.median(idata[f"{col}_{nn}_d2"]))
                            _idata.append(
                                np.max(idata[f"{col}_{nn}_d2"]) -
                                np.min(idata[f"{col}_{nn}_d2"]))
                            _idata.append(
                                np.max(idata[f"{col}_{nn}_d2"]) -
                                np.mean(idata[f"{col}_{nn}_d2"]))
                            _idata.append(
                                np.max(idata[f"{col}_{nn}_d2"]) -
                                np.median(idata[f"{col}_{nn}_d2"]))
                            _idata.append(
                                np.mean(idata[f"{col}_{nn}_d2"]) -
                                np.median(idata[f"{col}_{nn}_d2"]))
                            _idata.append(np.std(idata[f"{col}_{nn}_d2"]))

                        _idata.append(np.sum(idata[f"{col}"]))
                        _idata.append(np.max(idata[f"{col}"]))
                        _idata.append(np.mean(idata[f"{col}"]))
                        _idata.append(np.min(idata[f"{col}"]))
                        _idata.append(np.median(idata[f"{col}"]))
                        _idata.append(
                            np.max(idata[f"{col}"]) - np.min(idata[f"{col}"]))
                        _idata.append(
                            np.max(idata[f"{col}"]) - np.mean(idata[f"{col}"]))
                        _idata.append(
                            np.max(idata[f"{col}"]) -
                            np.median(idata[f"{col}"]))
                        _idata.append(
                            np.mean(idata[f"{col}"]) -
                            np.median(idata[f"{col}"]))
                        _idata.append(np.std(idata[f"{col}"]))

                    L = len(_idata)
                    if istar in dlist:
                        _idata.append(dlist[istar])
                    else:
                        _idata.append(dv["normal"])

                    alldata.append(_idata)
                except Exception as e:
                    print(e)
# 26.

cols = ["Starid"] + [f"Var{i}" for i in range(L - 1)] + ["Target"]
X = pd.DataFrame(alldata, columns=cols)
X.fillna(-999999, inplace=True)
print(X.shape)

tabn = "../user_data/tmp_data/D_{sample}.pkl"
X.to_pickle(tabn)

X = pd.read_pickle(tabn)
_X = X[X.columns.drop(["Starid", "Target"])]
print(X.shape)
print(pd.value_counts(X["Target"]))
print(pd.value_counts(X["Target"]) / pd.value_counts(X["Target"]).sum())

badr = 1 - (
    pd.value_counts(X["Target"]) / pd.value_counts(X["Target"]).sum())[0]
print(badr)

from sklearn.ensemble import IsolationForest

clf = IsolationForest(
    behaviour='new',
    #     max_samples=10,
    n_estimators=1000,
    contamination=badr,
    bootstrap=True,
    n_jobs=-1,
)
clf.fit(_X)
X["Cluster1"] = clf.predict(_X)
print(pd.crosstab(X["Cluster1"], X["Target"]))

from sklearn.cluster import MeanShift, estimate_bandwidth
bandwidth = estimate_bandwidth(_X, quantile=0.3, n_samples=300, n_jobs=-1)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(_X)
labels = ms.labels_
X["Cluster2"] = labels
print(pd.crosstab(X["Cluster2"], X["Target"]))

from sklearn.cluster import MeanShift, estimate_bandwidth
bandwidth = estimate_bandwidth(_X, quantile=0.1, n_samples=300, n_jobs=-1)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(_X)
labels = ms.labels_
X["Cluster3"] = labels
print(pd.crosstab(X["Cluster3"], X["Target"]))

from sklearn.cluster import MeanShift, estimate_bandwidth
bandwidth = estimate_bandwidth(_X, quantile=0.05, n_samples=300, n_jobs=-1)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(_X)
labels = ms.labels_
X["Cluster4"] = labels
print(pd.crosstab(X["Cluster4"], X["Target"]))

from sklearn.cluster import Birch
birch = Birch(n_clusters=40)
labels = birch.fit_predict(_X)
X["Cluster5"] = labels
print(pd.crosstab(X["Cluster5"], X["Target"]))

__X = X[[
    "Starid", "Cluster1", "Cluster2", "Cluster3", "Cluster4", "Cluster5",
    "Target"
]]
__X["K"] = [
    f"{k1}-{k2}-{k3}-{k4}-{k5}" for k1, k2, k3, k4, k5 in zip(
        __X["Cluster1"], __X["Cluster2"], __X["Cluster3"], __X["Cluster4"],
        __X["Cluster5"])
]
rule = pd.crosstab(__X["K"], __X["Target"])
print(rule.head(10))

M = 10
lF = rule[(rule[1] > 0) & (rule[0] <= M)]
lM = rule[(rule[2] > 0) & (rule[0] <= M)]

with open(f"../prediction_result/result.csv", "w") as f:
    for istar in __X[__X["K"].isin([l for l in lF.index])]["Starid"]:
        f.write(f"{istar},flare star\n")

    for istar in __X[__X["K"].isin([l for l in lM.index])]["Starid"]:
        f.write(f"{istar},microlensing\n")
