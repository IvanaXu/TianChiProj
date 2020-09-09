应用一个业务概念，所谓```贷款等级```指贷款申请前风险模型评分。那么只要模型还靠谱，其结果就有一定区分度。
于是我们使用变量：
```
subGrade,贷款等级之子级
```
并在训练集上获取坏账率
```
isDefault req        yp
34         G5  0.451163
33         G4  0.478029
32         G3  0.480573
29         F5  0.517007
31         G2  0.519090
28         F4  0.522560
30         G1  0.533826
27         F3  0.543193
26         F2  0.544009
25         F1  0.573502
24         E5  0.580839
23         E4  0.597757
22         E3  0.612540
21         E2  0.623097
20         E1  0.644767
19         D5  0.665265
18         D4  0.677137
17         D3  0.695985
16         D2  0.702428
15         D1  0.722018
14         C5  0.738451
13         C4  0.749887
12         C3  0.775424
11         C2  0.793108
10         C1  0.808640
9          B5  0.834351
8          B4  0.851361
7          B3  0.870761
6          B2  0.887738
5          B1  0.897079
4          A5  0.914601
3          A4  0.932779
2          A3  0.944118
1          A2  0.954303
0          A1  0.968081
```
套用之。

以下代码：
```python
#
import numpy as np
import pandas as pd

dt =  "/Users/ivan/Desktop/ALL/Data/TestRisk"
dtrai = pd.read_csv(f"{dt}/train.csv")
dtest = pd.read_csv(f"{dt}/testA.csv")

dtrai["req"] = dtrai.subGrade
dtest["req"] = dtest.subGrade

print(pd.value_counts(dtrai.isDefault))

_ = pd.crosstab(dtrai.req, dtrai.isDefault)
_["yp"] = _[0]/(_[0]+_[1])
_.reset_index(inplace=True)
_.sort_values(by="yp", inplace=True)
print(_[["req", "yp"]])

_r = pd.merge(dtest[["id", "req"]], _[["req", "yp"]], on="req", how="left")
_r["isDefault"] = [round(_,6) for _ in _r["yp"]]
_r.sort_values(by="id", inplace=True)
_r[["id", "isDefault"]].to_csv(f"{dt}/outs/submit.csv", index=None)
print(_r[["id", "isDefault"]].head())
```

