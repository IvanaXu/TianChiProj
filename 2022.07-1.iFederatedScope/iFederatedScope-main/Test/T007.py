
import sys
import numpy as np

if len(sys.argv) > 1:
    datp = "/root/proj/data/CIKM22Competition/"
else:
    datp = "/Users/ivan/Library/CloudStorage/OneDrive-个人/Code/CIKM2022/data/CIKM22Competition/"

import torch


def view(cid):
    setall = set()
    for data_type in ["train", "test", "val"]:
        import time
        time0 = time.time()
        _data = torch.load(f"{datp}/{cid}/{data_type}.pt")
        time1 = time.time() - time0
        setl = set()
        for idata in _data:
            for jdata in np.array(idata.edge_index):
                for kdata in jdata:
                    setl.add(kdata)
        print(data_type, setl)
        print(np.max(list(setl)), np.min(list(setl)))
        print(f"{time1:.6f}s")

        if len(setall) == 0:
            setall = setl
        else:
            setall = setall & setl
    print()
    print(setall)
    print(np.max(list(setall)), np.min(list(setall)))
    assert setall == set([i for i in range(np.max(list(setall))+1)])


for cid in range(1, 13+1):
    print("\n", cid)
    view(cid)
