# -*-coding:utf-8-*-
# @auth ivan
# @time 20191111 16:39:32 
# @goal r0file neg

import os
test = ["path\n"]

for i, _, k in os.walk("Cervical_Cancer/out/test"):
    for t in k:
        p = i + os.path.sep + t
        if "jpg" in p and not t.startswith("."):
            test.append(p+"\n")
# print(test)

with open("Cervical_Cancer/list/test.list", "w") as f:
    for i in test:
        f.write(i)

