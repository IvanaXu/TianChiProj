# -*-coding:utf-8-*-
# @auth ivan
# @time 20191111 16:39:32 
# @goal r0file

import os

pos = []

for i, _, k in os.walk("Cervical_Cancer/out/pos"):
    for t in k:
        p = i + os.path.sep + t
        if "jpg" in p and not t.startswith("."):
            pos.append(p+"\n")

with open("Cervical_Cancer/list/pos.list", "w") as f:
    for i in pos:
        f.write(i)

