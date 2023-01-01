
# -*-coding:utf-8-*-
# @auth ivan
# @time 2020-03-20 20:40:50
# @goal v0

import sys
import datetime
import calendar
import pandas as pd
import xgboost as xgb
import h5py
import zipfile


if sys.argv[-1] == "pro":
    inp_path = "/tcdata"
else:
    inp_path = "/data/code/gproj/code/PAKDD2020/project/data"
inp_name = "/disk_sample_smart_log_round2"
bst = xgb.Booster(model_file="mdl/m002.model")


def month_end(get_date):
    dat1 = datetime.datetime.strptime(get_date, '%Y%m')
    y, m, d = dat1.year, dat1.month, dat1.day
    if m == 12:
        m = 1
        y += 1
    else:
        m += 1
    dat2 = datetime.datetime(y, m, d)

    edy1 = calendar.monthrange(dat1.year, dat1.month)[1]
    edy2 = calendar.monthrange(dat2.year, dat2.month)[1]

    return dat1.year, dat1.month, edy1, dat2.year, dat2.month, edy2


result_l = []
for i_mth in ["201808", "201809"]:
    d_smart_dl = pd.read_hdf(f"mdl/d_smart_dl.h5", key="data")
    d_smart_dl = pd.DataFrame(d_smart_dl)
    smart_l = list(d_smart_dl.index)
    col = ["manufacturer", "model", "serial_number", "dt"] + smart_l

    l_mth = month_end(i_mth)
    # print(l_mth)

    bT = pd.read_csv(f"mdl/disk_sample_fault_tag_all.csv")
    bT["mk"] = [f"{i}{j}{k}" for i, j, k in zip(bT["manufacturer"], bT["model"], bT["serial_number"])]
    bT['fault_time1'] = bT['fault_time'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d'))

    bT0 = bT.copy()
    bT0 = bT0[bT0["fault_time1"] <= datetime.datetime(l_mth[0], l_mth[1], l_mth[2])]
    for i in range(0, 7):
        bT0[f"tag{i}"] = bT0["tag"].apply(lambda x: 1 if x == i else 0)
    bT0 = bT0[["mk"] + [f"tag{i}" for i in range(0, 7)]].groupby(by="mk").sum()
    bT0.reset_index(inplace=True)
    # print(bT0.head())

    l_data = []
    st = int(i_mth)*100+1 if i_mth == "201809" else int(i_mth)*100+11
    N = 30 if i_mth == "201809" else 21
    for i in range(st, st+N):
        # print(i)
        data = pd.read_csv(
            f"{inp_path}{inp_name}/disk_sample_smart_log_{i}_round2.csv",
            usecols=col,
        )
        l_data.append(data)
    data = pd.DataFrame(pd.concat(l_data))

    data["mk"] = [f"{i}{j}{k}" for i, j, k in zip(data["manufacturer"], data["model"], data["serial_number"])]
    data["dt1"] = data["dt"].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d"))
    data["num"] = 1

    # AVG
    data_avg = data[["mk"] + smart_l].groupby(by="mk").mean()
    rename_l = {"num": "x_avg"}
    for i, v in enumerate(smart_l):
        rename_l[v] = f"x_avg_{i}"
    data_avg.rename(columns=rename_l, inplace=True)
    data_avg.reset_index(inplace=True)
    data_avg.fillna(-1, inplace=True)

    # MIN
    data_min = data[["mk"] + smart_l].groupby(by="mk").min()
    rename_l = {"num": "x_min"}
    for i, v in enumerate(smart_l):
        rename_l[v] = f"x_min_{i}"
    data_min.rename(columns=rename_l, inplace=True)
    data_min.reset_index(inplace=True)

    # P50
    data_p50 = data[["mk"] + smart_l].groupby(by="mk").median()
    rename_l = {"num": "x_p50"}
    for i, v in enumerate(smart_l):
        rename_l[v] = f"x_p50_{i}"
    data_p50.rename(columns=rename_l, inplace=True)
    data_p50.reset_index(inplace=True)

    # MAX
    data_max = data[["mk", "dt1"] + smart_l].groupby(by="mk").max()
    rename_l = {"num": "x_max"}
    for i, v in enumerate(smart_l):
        rename_l[v] = f"x_max_{i}"
    data_max.rename(columns=rename_l, inplace=True)
    data_max.reset_index(inplace=True)

    # STD
    data_std = data[["mk"] + smart_l].groupby(by="mk").std()
    rename_l = {"num": "x_std"}
    for i, v in enumerate(smart_l):
        rename_l[v] = f"x_std_{i}"
    data_std.rename(columns=rename_l, inplace=True)
    data_std.reset_index(inplace=True)
    data_std.fillna(-1, inplace=True)

    d_out = data_avg
    d_out = pd.merge(d_out, data_min, on="mk", how="left")
    d_out = pd.merge(d_out, data_p50, on="mk", how="left")
    d_out = pd.merge(d_out, data_max, on="mk", how="left")
    d_out = pd.merge(d_out, data_std, on="mk", how="left")
    d_out = pd.merge(d_out, bT0, on="mk", how="left")
    # d_out = pd.merge(d_out, bT1, on="mk", how="left")

    for i, v in enumerate(smart_l):
        d_out[f"x_max_d_min_{i}"] = d_out[f"x_max_{i}"] - d_out[f"x_min_{i}"]

        d_out[f"x_max_d_avg_{i}"] = d_out[f"x_max_{i}"] - d_out[f"x_avg_{i}"]
        d_out[f"x_max_p_avg_{i}"] = d_out[f"x_min_{i}"] + d_out[f"x_avg_{i}"]
        d_out[f"x_max_d_p50_{i}"] = d_out[f"x_max_{i}"] - d_out[f"x_p50_{i}"]
        d_out[f"x_max_p_p50_{i}"] = d_out[f"x_min_{i}"] + d_out[f"x_p50_{i}"]
        d_out[f"x_avg_p_1std_{i}"] = d_out[f"x_avg_{i}"] + 1 * d_out[f"x_std_{i}"]
        d_out[f"x_avg_d_1std_{i}"] = d_out[f"x_avg_{i}"] - 1 * d_out[f"x_std_{i}"]
        d_out[f"x_avg_p_2std_{i}"] = d_out[f"x_avg_{i}"] + 2 * d_out[f"x_std_{i}"]
        d_out[f"x_avg_d_2std_{i}"] = d_out[f"x_avg_{i}"] - 2 * d_out[f"x_std_{i}"]
        d_out[f"x_avg_p_3std_{i}"] = d_out[f"x_avg_{i}"] + 3 * d_out[f"x_std_{i}"]
        d_out[f"x_avg_d_3std_{i}"] = d_out[f"x_avg_{i}"] - 3 * d_out[f"x_std_{i}"]

    d_out.fillna(0, inplace=True)

    var_l = d_out.columns.drop(["mk", "dt1"])
    with h5py.File('mdl/value.h5', "r") as f:
        Nr = f["Nr"][0]
    print(f"CutOFF:{Nr}")

    predict = bst.predict(xgb.DMatrix(d_out[var_l]))

    result = d_out
    result["maybe"] = [1 if i >= Nr else 0 for i in predict]
    result = result[result["maybe"] == 1]

    result = pd.merge(result[["mk", "maybe"]], data_max[["mk", "dt1"]], on="mk", how="left")

    result.sort_values(by=["mk", "dt1"], inplace=True)
    print(result.shape)
    result.drop_duplicates(subset=['mk'], keep='first', inplace=True)
    print(result.shape)

    result["manufacturer"] = result["mk"].apply(lambda x: x[0])
    result["model"] = result["mk"].apply(lambda x: x[1])
    result["serial_number"] = result["mk"].apply(lambda x: x[2:])
    result_l.append(result)


result = pd.concat(result_l)
print("__________")
print(result.shape)
print(pd.value_counts(result["dt1"]).sort_index())

result[["manufacturer", "model", "serial_number", "dt1"]].to_csv(
    f"predictions.csv",
    index=False, header=False
)

with zipfile.ZipFile("result.zip", "w") as z:
    z.write("predictions.csv", compress_type=zipfile.ZIP_DEFLATED)



