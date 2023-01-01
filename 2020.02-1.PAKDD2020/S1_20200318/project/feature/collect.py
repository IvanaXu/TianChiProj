# -*-coding:utf-8-*-
# @auth ivan
# @time 2020-03-17 02:15:43
# @goal collect

import datetime
import calendar
import pandas as pd
from tqdm import tqdm

mth_l = [
    "201707", "201708", "201709", "201710", "201711", "201712",
    "201801", "201802", "201803", "201804", "201805", "201806",
    # "201807",
]

d_smart_dl = pd.read_hdf(f"../user_data/tmp_data/d_smart_dl.h5", key="data")
d_smart_dl = pd.DataFrame(d_smart_dl)
smart_l = list(d_smart_dl.index)
col = ["manufacturer", "model", "serial_number", "dt"] + smart_l

bT = pd.read_csv(f"../data/round1_train/disk_sample_fault_tag.csv")
bT["mk"] = [f"{i}{j}{k}" for i, j, k in zip(bT["manufacturer"], bT["model"], bT["serial_number"])]
bT['fault_time1'] = bT['fault_time'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d'))


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


print(month_end("201801"), month_end("201812"))

for i_mth in tqdm(mth_l):
    l_mth = month_end(i_mth)

    bT0 = bT.copy()
    bT0 = bT0[bT0["fault_time1"] <= datetime.datetime(l_mth[0], l_mth[1], l_mth[2])]
    for i in range(0, 7):
        bT0[f"tag{i}"] = bT0["tag"].apply(lambda x: 1 if x == i else 0)
    bT0 = bT0[["mk"] + [f"tag{i}" for i in range(0, 7)]].groupby(by="mk").sum()
    bT0.reset_index(inplace=True)

    bT1 = bT.copy()
    bT1 = bT1[bT1["fault_time1"] > datetime.datetime(l_mth[0], l_mth[1], l_mth[2])]
    bT1 = bT1[bT1["fault_time1"] <= datetime.datetime(l_mth[3], l_mth[4], l_mth[5])]
    bT1["bad"] = 1
    bT1.drop_duplicates("mk", inplace=True)
    bT1 = bT1[["mk", "bad"]]

    data = pd.read_csv(
        f"../data/round1_train/disk_sample_smart_log_{i_mth}.csv",
        usecols=col,
    )
    data["mk"] = [f"{i}{j}{k}" for i, j, k in zip(data["manufacturer"], data["model"], data["serial_number"])]
    data["dt1"] = data["dt"].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d"))
    data["num"] = 1

    #
    # SUM
    data_sum = data[["mk", "num"] + smart_l].groupby(by="mk").sum()
    rename_l = {"num": "x_a"}
    for i, v in enumerate(smart_l):
        data_sum[v] = data_sum[v]/data_sum["num"]
        rename_l[v] = f"x_a_{i}"
    data_sum.rename(columns=rename_l, inplace=True)
    data_sum.reset_index(inplace=True)

    # MIN
    data_min = data[["mk"] + smart_l].groupby(by="mk").min()
    rename_l = {"num": "x_s"}
    for i, v in enumerate(smart_l):
        rename_l[v] = f"x_s_{i}"
    data_min.rename(columns=rename_l, inplace=True)
    data_min.reset_index(inplace=True)

    # MEDIAN
    data_med = data[["mk"] + smart_l].groupby(by="mk").median()
    rename_l = {"num": "x_m"}
    for i, v in enumerate(smart_l):
        rename_l[v] = f"x_m_{i}"
    data_med.rename(columns=rename_l, inplace=True)
    data_med.reset_index(inplace=True)

    # MAX
    data_max = data[["mk"] + smart_l].groupby(by="mk").max()
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
    d_out = pd.merge(d_out, bT1, on="mk", how="left")
    d_out.fillna(0, inplace=True)
    print(i_mth, d_out.shape, d_out.head())

    with pd.HDFStore(f"../user_data/tmp_data/d_out_{i_mth}.h5", 'w', complevel=4, complib='blosc') as f:
        f.put(key='data', value=d_out, format='table')



