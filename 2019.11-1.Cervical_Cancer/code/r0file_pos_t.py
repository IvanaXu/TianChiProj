# -*-coding:utf-8-*-
# @auth ivan
# @time 20191111 16:39:32 
# @goal r0file

import os

pos_t = []

for i, _, k in os.walk("Cervical_Cancer/out/pos_t"):
    for t in k:
        p = i + os.path.sep + t
        if "jpg" in p and not t.startswith("."):
            pos_t.append(p+"\n")
# print(pos_t)

with open("Cervical_Cancer/list/pos_t.list", "w") as f:
    for i in pos_t:
        f.write(i)

