# -*-coding:utf-8-*-
# @auth ivan
# @time 20191119 14:55:32 
# @goal test

"""
T2019_600.json
[{"x": 22890, "y": 3877, "w": 396, "h": 255，"p": 0.94135}, 
{"x": 20411, "y": 2260, "w": 8495, "h": 7683，"p": 0.67213}, 
{"x": 26583, "y": 7937, "w": 172, "h": 128，"p": 0.73228}, 
{"x": 2594, "y": 18627, "w": 1296, "h": 1867，"p": 0.23699}]
"""

import os
import sys
print(sys.argv[1])

para = str(sys.argv[1])
rf = open("Cervical_Cancer/result/"+para+".log", "r")
result = {}

if not os.path.exists("Cervical_Cancer/result/"+para):
    os.system("mkdir Cervical_Cancer/result/"+para)

for i in rf:
    i = i.strip("\n")
    i = i.replace("Cervical_Cancer/out/test/", "")
    i = i.replace(".jpg", "")
    i = i.replace("/test_", ",")
    i = i.replace("_", ",")
    i = i.replace("T2019,", "T2019_")
    i = i.split(",")

    if len(i) == 6:
        [t1, t2, t3, t4, t5, t6] = i
        t1 = t1 + ".json"
        t2, t3, t4, t5 = int(t2), int(t3), int(t4), int(t5)
        t6 = float(t6)
        print(t1, t2, t3, t4, t5, t6)

        if t1 not in result:
            result[t1] = [{"x": t2, "y": t3, "w": t4, "h": t5, "p": t6}]
        else:
            result[t1] = result[t1] + [{"x": t2, "y": t3, "w": t4, "h": t5, "p": t6}]

for i, j in result.items():
    j = str(j).replace("'",'"')
    print(i, j)
    with open("Cervical_Cancer/result/"+para+"/"+i,"w") as f:
        f.write(j)

pmp = "Cervical_Cancer/"
for testi in ["test_0", "test_1", "test_2", "test_3"]:
    pmpi = pmp + testi
    for i, _, k in os.walk(pmpi):
        for t in k:
            p = i+os.path.sep+t
            p = p.replace(pmpi+"/","").replace(".kfb",".json")
            p = "Cervical_Cancer/result/"+para+"/"+p
            if not os.path.exists(p):
                with open(p, "w") as f:
                    f.write("[]")
                print(p)
os.system("zip -q -r @_@.zip @_@/".replace("@_@", "Cervical_Cancer/result/"+para))

