# -*-coding:utf-8-*-
# @auth ivan
# @time 2020-03-17 04:15:43
# @goal build_data

import pandas as pd
from tqdm import tqdm

# i_model = "A1"
i_model = "A2"
ns = 300

mth_l = [
    "201707", "201708", "201709", "201710", "201711", "201712",
    "201801", "201802", "201803", "201804", "201805",
    # "201806", "201807",
]

data_t01 = pd.DataFrame([])

for i_mth in tqdm(mth_l):
    print(i_mth)

    data = pd.read_hdf(f"../user_data/tmp_data/d_out_{i_mth}.h5", key="data")
    data = pd.DataFrame(data)
    data["mk2"] = data["mk"].apply(lambda x: str(x)[:2])
    data = data[data["mk2"] == i_model]
    data.drop(columns="mk2", inplace=True)

    t_num = pd.value_counts(data["bad"])
    t_num0, t_num1 = t_num[0], t_num[1]
    print(t_num0, t_num1)

    data_0 = data[data["bad"] == 0].sample(min(t_num0, t_num1*ns))
    data_1 = data[data["bad"] == 1].sample(t_num1)

    data_t01 = pd.concat([data_t01, data_0, data_1])

print(data_t01.shape)
print(pd.value_counts(data_t01["bad"]))

with pd.HDFStore(f"../user_data/tmp_data/data_t01.h5", 'w', complevel=4, complib='blosc') as f:
    f.put(key='data', value=data_t01, format='table')


data_v01 = pd.read_hdf(f"../user_data/tmp_data/d_out_201806.h5", key="data")
data_v01 = pd.DataFrame(data_v01)
data_v01["mk2"] = data_v01["mk"].apply(lambda x: str(x)[:2])
data_v01 = data_v01[data_v01["mk2"] == i_model]
data_v01.drop(columns="mk2", inplace=True)

print(data_v01.shape)
print(pd.value_counts(data_v01["bad"]))
with pd.HDFStore(f"../user_data/tmp_data/data_v01.h5", 'w', complevel=4, complib='blosc') as f:
    f.put(key='data', value=data_v01, format='table')



