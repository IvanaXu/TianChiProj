#
from tqdm import tqdm
import pandas as pd
N = 100
dp = "/Users/ivan/Desktop/ALL/Code/PyProduct/GitProj/gproj/code/PAKDD2020/project/data/"
r1_a = f"{dp}round1_testA/disk_sample_smart_log_test_a.csv"
r1_b = f"{dp}round1_testB/disk_sample_smart_log_test_b.csv"
out = "disk_sample_smart_log_round2"

data_a = pd.read_csv(r1_a)
data_b = pd.read_csv(r1_b)
data = pd.concat([data_a, data_b])

l_time = [
    i for i in
    pd.value_counts(data["dt"]).index
    # if i != 20180831
]

for i in tqdm(l_time):
    print(i)

    data_ = data[data["dt"] == i].sample(N)

    data_["dt"] = i
    data_.to_csv(f"{dp}{out}/disk_sample_smart_log_{i}_round2.csv", index=False)

    data_["dt"] = i+100
    data_.to_csv(f"{dp}{out}/disk_sample_smart_log_{i+100}_round2.csv", index=False)



