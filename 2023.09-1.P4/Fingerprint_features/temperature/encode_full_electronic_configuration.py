# Import panda to deal with *.csv file
import pandas as pd

"""
# 从csv文件中读取不同原子轨道的电子占用和金属的氧化态
# 同时读取需要预测的temperature和time
"""

#############

df = pd.read_csv("finger_example.csv")  # read the *csv file

l = 100

##Here We just simply keep on reading the electronic occupancy of the different atomic
##orbital from the csv file. The metal oxidation state is also read.

s1 = []
tem = df["1s"]

for i in range(0, l):
    c = [tem[i]]
    s1.append(c)

s2 = []
tem = df["2s"]

for i in range(0, l):
    c = [tem[i]]
    s2.append(c)

s3 = []
tem = df["3s"]


for i in range(0, l):
    c = [tem[i]]
    s3.append(c)

s4 = []
tem = df["4s"]


for i in range(0, l):
    c = [tem[i]]
    s4.append(c)

s5 = []
tem = df["5s"]


for i in range(0, l):
    c = [tem[i]]
    s5.append(c)

s6 = []
tem = df["6s"]


for i in range(0, l):
    c = [tem[i]]
    s6.append(c)


p2 = []
tem = df["2p"]


for i in range(0, l):
    c = [tem[i]]
    p2.append(c)

p3 = []
tem = df["3p"]


for i in range(0, l):
    c = [tem[i]]
    p3.append(c)

p4 = []
tem = df["4p"]


for i in range(0, l):
    c = [tem[i]]
    p4.append(c)

p5 = []
tem = df["5p"]


for i in range(0, l):
    c = [tem[i]]
    p5.append(c)


d3 = []
tem = df["3d"]


for i in range(0, l):
    c = [tem[i]]
    d3.append(c)

d4 = []
tem = df["4d"]


for i in range(0, l):
    c = [tem[i]]
    d4.append(c)


d5 = []
tem = df["5d"]


for i in range(0, l):
    c = [tem[i]]
    d5.append(c)


f4 = []
tem = df["4f"]

for i in range(0, l):
    c = [tem[i]]
    f4.append(c)

o = []
tem = df["oxidation_state"]

for i in range(0, l):
    c = [tem[i]]
    o.append(c)


temp = []
tem = df["temperature"]


for i in range(0, l):
    c = [tem[i]]

    temp.append(c)


time = []
tem = df["time"]


for i in range(0, l):
    c = [tem[i]]

    time.append(c)
