# -*-coding:utf-8-*-
# @auth ivan
# @time 2020-03-17 00:15:43
# @goal feature

import pandas as pd
from tqdm import tqdm

mth_l = [
    "201707", "201708", "201709", "201710", "201711", "201712",
    "201801", "201802", "201803", "201804", "201805", "201806",
    "201807",
]

smart_l = []
for i in range(1, 256):
    smart_l.append(f"smart_{i}_normalized")
    # smart_l.append(f"smart_{i}raw")

smart_dl = {}
for col in smart_l:
    smart_dl[col] = 0

for mth in tqdm(mth_l):
    data = pd.read_csv(
        f"../data/round1_train/disk_sample_smart_log_{mth}.csv",
        usecols=smart_l, chunksize=1000000,
    )
    for i_data in data:
        print("-", end="")
        for col in smart_l:
            smart_dl[col] = smart_dl[col] + ((i_data[col].nunique() == 1) | ((-i_data[col].isnull()).sum() == 0))

d_smart_dl = pd.DataFrame.from_dict(smart_dl, orient="index")
d_smart_dl = d_smart_dl[d_smart_dl[0] < d_smart_dl[0].max()]

with pd.HDFStore(f"../user_data/tmp_data/d_smart_dl.h5", 'w', complevel=4, complib='blosc') as f:
    f.put(key='data', value=d_smart_dl, format='table')



