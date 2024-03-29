# -*-coding:utf-8-*-
# @auth ivan
# @time 20191111 16:39:32 
# @goal r0file neg

import os
neg = []

for i, _, k in os.walk("Cervical_Cancer/out/neg"):
    for t in k:
        p = i + os.path.sep + t
        if "jpg" in p and not t.startswith("."):
            neg.append(p+"\n")
# print(neg)

with open("Cervical_Cancer/list/neg.list", "w") as f:
    for i in neg:
        f.write(i)

