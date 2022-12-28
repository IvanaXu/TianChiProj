import sys
if len(sys.argv) > 1:
    datp = "/root/proj/data/CIKM22Competition/"
else:
    datp = "/Users/ivan/Library/CloudStorage/OneDrive-个人/Code/CIKM2022/data/CIKM22Competition/"

#
# pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install torch_geometric torch_sparse torch_scatter -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install numpy pandas sklearn -i https://pypi.tuna.tsinghua.edu.cn/simple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


for cid in range(1, 9):
    #
    data = torch.load(f"{datp}/{cid}/train.pt")
    coll = data[0].x.shape[1]
    cols = [f"x{i}" for i in range(coll)]

    xdata, ydata = [], []
    for idata in data:
        xdata.append(np.mean(np.array(idata.x), axis=0))
        ydata.append(np.array(idata.y)[0][0])

    _data = pd.DataFrame(xdata, columns=cols)
    _data["y"] = ydata
    # print(pd.value_counts(_data["y"]))

    #
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(n_jobs=-1)
    model.fit(_data[cols], _data["y"])

    #
    from sklearn.metrics import accuracy_score
    print(f""">>> {cid} / Train Error rate: {1 - accuracy_score(_data["y"], model.predict(_data[cols])):.6f}""")

    #
    data = torch.load(f"{datp}/{cid}/val.pt")

    xdata, ydata = [], []
    for idata in data:
        xdata.append(np.mean(np.array(idata.x), axis=0))
        ydata.append(np.array(idata.y)[0][0])

    _data = pd.DataFrame(xdata, columns=cols)
    _data["y"] = ydata
    # print(pd.value_counts(_data["y"]))
    print(f""">>> {cid} / Valal Error rate: {1 - accuracy_score(_data["y"], model.predict(_data[cols])):.6f}""")
    print()
