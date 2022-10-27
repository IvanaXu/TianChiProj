#
import numpy as np
import pandas as pd

dt = "/Users/ivan/Desktop/ALL/Data/TestRisk"

dtrai = pd.read_csv(f"{dt}/train.csv")
dtest = pd.read_csv(f"{dt}/testA.csv")

dtrai["req"] = dtrai.subGrade
dtest["req"] = dtest.subGrade

print(pd.value_counts(dtrai.isDefault))

_ = pd.crosstab(dtrai.req, dtrai.isDefault)
_["yp"] = _[1]/(_[0]+_[1])
_.reset_index(inplace=True)
_.sort_values(by="yp", inplace=True)
print(_[["req", "yp"]])

_r = pd.merge(dtest[["id", "req"]], _[["req", "yp"]], on="req", how="left")
_r["isDefault"] = [round(_,6) for _ in _r["yp"]]
_r.sort_values(by="id", inplace=True)
_r[["id", "isDefault"]].to_csv(f"{dt}/outs/submit.csv", index=None)
print(_r[["id", "isDefault"]].head())




