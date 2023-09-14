from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import os
import pylab

from sklearn.metrics import mean_absolute_error

"""
汇总测试集的十折交叉验证的结果
并计算所有预测结果的mae和r2指标
"""


def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def SMAPE(y_true, y_pred):
    return (
        2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
    )


with open("./predictions_rawdata/real_all_test.txt", "w") as f:
    for i in range(1, 11):
        with open(f"./predictions_rawdata/y_real_{i}_test.txt") as g:
            f.write(g.read())

with open("./predictions_rawdata/pred_all_test.txt", "w") as f:
    for i in range(1, 11):
        with open(f"./predictions_rawdata/y_RFR_{i}_test.txt") as g:
            f.write(g.read())

real = pylab.loadtxt("./predictions_rawdata/real_all_test.txt")
pred = pylab.loadtxt("./predictions_rawdata/pred_all_test.txt")

mae = round(mean_absolute_error(real, pred), 3)

r2_score = round(r2_score(real, pred), 3)

mape = round(MAPE(real, pred), 3)
smape = round(SMAPE(real, pred), 3)

print(mae, r2_score, mape, smape)
