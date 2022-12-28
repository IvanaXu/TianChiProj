# -*-coding:utf-8-*-
# @date 2022-09-10 03:32:09
# @auth Ivan
# @goal Feature To Xdata

import os
from config import *
import pandas as pd
from tqdm import tqdm

datp = "/Users/ivan/Desktop/Data/CIKM2022/CIKM22Competition/"
pklp = f"{datp}/X"

for [cid, ctop, cNLP] in tqdm([
    [1, 111, ""],
    [2, 29, "NLP"],
    [3, 105, ""],
    [4, 22, "NLP"],
    [5, 10, ""],
    [6, 99, ""],
    [7, 91, ""],
    [8, 63, ""],
    [9, 36, ""],
    [10, 11, ""],
    [11, 48, ""],
    [12, 34, ""],
    [13, 28, ""],
]):
    for cdata_type in ["train", "test", "val"]:
        cdata = get_data(datp, pklp, cid, cdata_type, ctop, cNLP)
        ddata = pd.read_pickle(f"{pklp}/X-{cid}-{cdata_type}.pkl", compression="zip")
        assert cdata[["data_index", "y"] + COLS].shape == ddata[["data_index", "y"] + COLS].shape

os.system('say "i finish the job"')
