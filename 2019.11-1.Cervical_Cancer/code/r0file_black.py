# -*-coding:utf-8-*-
# @auth ivan
# @time 20191120 14:13:00
# @goal black

import os
black = []

for i, _, k in os.walk("Cervical_Cancer/out/black"):
    for t in k:
        p = i + os.path.sep + t
        if "jpg" in p and not t.startswith("."):
            black.append(p+"\n")

with open("Cervical_Cancer/list/black.list", "w") as f:
    for i in black:
        f.write(i)



