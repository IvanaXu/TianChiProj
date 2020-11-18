# 程序入口
sh main.sh

# 核心思路
1、构建变量
```python
for col in [
    "jd", "jd_c", "magnorm", "mage", "magnorm_m0", "magnorm_m1",
    "magnorm_dmean", "magnorm_m0_dmean", "magnorm_m1_dmean"
]:
    for nn in range(1, 60):
        dvar.append(f"{col}_{nn}_d1_np.sum")
        dvar.append(f"{col}_{nn}_d1_np.max")
        dvar.append(f"{col}_{nn}_d1_np.mean")
        dvar.append(f"{col}_{nn}_d1_np.min")
        dvar.append(f"{col}_{nn}_d1_np.median")
        dvar.append(f"{col}_{nn}_d1_np.max-np.min")
        dvar.append(f"{col}_{nn}_d1_np.max-np.mean")
        dvar.append(f"{col}_{nn}_d1_np.max-np.median")
        dvar.append(f"{col}_{nn}_d1_np.mean-np.median")
        dvar.append(f"{col}_{nn}_d1_np.std")
        
        dvar.append(f"{col}_{nn}_d2_np.sum")
        dvar.append(f"{col}_{nn}_d2_np.max")
        dvar.append(f"{col}_{nn}_d2_np.mean")
        dvar.append(f"{col}_{nn}_d2_np.min")
        dvar.append(f"{col}_{nn}_d2_np.median")
        dvar.append(f"{col}_{nn}_d2_np.max-np.min")
        dvar.append(f"{col}_{nn}_d2_np.max-np.mean")
        dvar.append(f"{col}_{nn}_d2_np.max-np.median")
        dvar.append(f"{col}_{nn}_d2_np.mean-np.median")
        dvar.append(f"{col}_{nn}_d2_np.std")
    
    dvar.append(f"{col}_np.sum")
    dvar.append(f"{col}_np.max")
    dvar.append(f"{col}_np.mean")
    dvar.append(f"{col}_np.min")
    dvar.append(f"{col}_np.median")
    dvar.append(f"{col}_np.max-np.min")
    dvar.append(f"{col}_np.max-np.mean")
    dvar.append(f"{col}_np.max-np.median")
    dvar.append(f"{col}_np.mean-np.median")
    dvar.append(f"{col}_np.std")
```
2、异常检测

引入孤立森林

3、聚类算法

引入`from sklearn.cluster import MeanShift, estimate_bandwidth`和`from sklearn.cluster import Birch`

4、结果整合

整合检测类别1-5，取出其中小量级群体作为输出结果。

5、结果展示

```
Target     0   1  2
Cluster1           
-1        16   0  1
 1        36  13  3

Target     0  1  2
Cluster2          
0         28  2  1
1          8  5  1
2         11  3  2
3          5  3  0

Target     0  1  2
Cluster3          
0         14  1  1
1         11  1  0
2          4  2  0
3          4  3  1
4          7  2  1
5          4  0  1
6          2  1  0
7          0  2  0
8          3  0  0
9          2  0  0
10         0  1  0
11         1  0  0

Target    0  1  2
Cluster4         
0         9  1  1
1         4  2  1
2         5  0  0
3         5  0  0
4         1  2  1
5         4  0  0
6         2  0  0
7         2  2  0
8         3  1  0
9         2  1  0
10        2  0  1
11        2  0  0
12        1  1  0
13        2  0  0
14        2  0  0
15        2  0  0
16        1  0  0
17        1  0  0
18        0  1  0
19        0  1  0
20        1  0  0
21        0  1  0
22        1  0  0

Target    0  1  2
Cluster5         
0         2  1  0
1         2  0  0
2         2  0  0
3         1  2  1
4         2  0  0
5         2  0  0
6         1  1  0
7         2  0  0
8         2  0  0
9         1  0  0
10        0  1  1
11        2  0  0
12        5  1  1
13        3  0  0
14        0  1  0
15        2  0  0
16        1  1  0
17        1  0  0
18        2  0  0
19        1  0  0
20        1  0  0
21        2  0  0
22        1  0  0
23        1  0  0
24        0  1  0
25        0  1  0
26        2  2  0
27        1  0  0
28        0  1  0
29        0  0  1
30        1  0  0
31        1  0  0
32        1  0  0
33        1  0  0
34        1  0  0
35        1  0  0
36        1  0  0
37        1  0  0
38        1  0  0
39        1  0  0

###
Target        0  1  2
K                    
-1-0-0-0-1    2  0  0
-1-0-0-0-12   4  0  0
-1-0-0-0-35   1  0  0
-1-0-0-2-34   1  0  0
-1-0-1-3-11   1  0  0
-1-0-1-6-39   1  0  0
-1-0-4-8-13   1  0  0
-1-2-2-15-33  1  0  0
-1-2-3-17-23  1  0  0
-1-2-5-10-29  0  0  1
```

