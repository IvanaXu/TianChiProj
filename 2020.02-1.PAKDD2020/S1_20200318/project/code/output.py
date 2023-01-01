import datetime
import calendar
import pandas as pd
import xgboost as xgb
import h5py

bst = xgb.Booster(model_file="../user_data/model_data/m001.model")

i_mth = "201808"
d_smart_dl = pd.read_hdf(f"../user_data/tmp_data/d_smart_dl.h5", key="data")
d_smart_dl = pd.DataFrame(d_smart_dl)
smart_l = list(d_smart_dl.index)
col = ["manufacturer", "model", "serial_number", "dt"] + smart_l


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


l_mth = month_end(i_mth)

bT = pd.read_csv(f"../data/round1_train/disk_sample_fault_tag.csv")
bT["mk"] = [f"{i}{j}{k}" for i, j, k in zip(bT["manufacturer"], bT["model"], bT["serial_number"])]
bT['fault_time1'] = bT['fault_time'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d'))

bT0 = bT.copy()
bT0 = bT0[bT0["fault_time1"] <= datetime.datetime(l_mth[0], l_mth[1], l_mth[2])]
for i in range(0, 7):
    bT0[f"tag{i}"] = bT0["tag"].apply(lambda x: 1 if x == i else 0)
bT0 = bT0[["mk"] + [f"tag{i}" for i in range(0, 7)]].groupby(by="mk").sum()
bT0.reset_index(inplace=True)

data = pd.read_csv(
    "../data/round1_testB/disk_sample_smart_log_test_b.csv",
    usecols=col,
)
data["mk"] = [f"{i}{j}{k}" for i, j, k in zip(data["manufacturer"], data["model"], data["serial_number"])]
data["dt1"] = data["dt"].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d"))
data["num"] = 1

data_sum = data[["mk", "num"] + smart_l].groupby(by="mk").sum()
rename_l = {"num": "x_a"}
for i, v in enumerate(smart_l):
    data_sum[v] = data_sum[v]/data_sum["num"]
    rename_l[v] = f"x_a_{i}"
data_sum.rename(columns=rename_l, inplace=True)
data_sum.reset_index(inplace=True)

data_min = data[["mk"] + smart_l].groupby(by="mk").min()
rename_l = {"num": "x_s"}
for i, v in enumerate(smart_l):
    rename_l[v] = f"x_s_{i}"
data_min.rename(columns=rename_l, inplace=True)
data_min.reset_index(inplace=True)

data_med = data[["mk"] + smart_l].groupby(by="mk").median()
rename_l = {"num": "x_m"}
for i, v in enumerate(smart_l):
    rename_l[v] = f"x_m_{i}"
data_med.rename(columns=rename_l, inplace=True)
data_med.reset_index(inplace=True)

data_max = data[["mk", "dt1"] + smart_l].groupby(by="mk").max()
rename_l = {"num": "x_l"}
for i, v in enumerate(smart_l):
    rename_l[v] = f"x_l_{i}"
data_max.rename(columns=rename_l, inplace=True)
data_max.reset_index(inplace=True)

d_out = data_sum
d_out = pd.merge(d_out, data_min, on="mk", how="left")
d_out = pd.merge(d_out, data_med, on="mk", how="left")
d_out = pd.merge(d_out, data_max, on="mk", how="left")
d_out = pd.merge(d_out, bT0, on="mk", how="left")
d_out.fillna(0, inplace=True)


#
var_l = d_out.columns.drop(["mk", "dt1"])
with h5py.File('../user_data/tmp_data/value.h5', "r") as f:
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
print(pd.value_counts(result["dt1"]))

result["manufacturer"] = result["mk"].apply(lambda x: x[0])
result["model"] = result["mk"].apply(lambda x: x[1])
result["serial_number"] = result["mk"].apply(lambda x: x[2:])

result[["manufacturer", "model", "serial_number", "dt1"]].to_csv(
    f"../prediction_result/predictions.csv",
    index=False, header=False
)

# 4.0864
# >> A,Cut:0.030000, F1:8.55154734
# >> S,Cut:0.030000, F1:3.77358491
# >> B,Cut:0.030000, F1:2.38568588



