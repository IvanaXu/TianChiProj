import os
import torch
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec

datp = "/Users/ivan/Desktop/Data/CIKM2022/CIKM22Competition/"
mdlp = f"{datp}/Server"
mdlf = f"{mdlp}/NLP.model"

os.system(f"rm -rf {mdlf}")
for cid in tqdm(range(1, 13+1)):
    _corpus = []
    for cdata_type in ["train", "val", "test"]:
        _data = torch.load(f"{datp}/{cid}/{cdata_type}.pt")
        # print(cdata_type, len(_data))
        for idata in _data:
            _corpus.append([str(i) for i in np.array(idata.edge_index[1])])

    if os.path.exists(mdlf):
        model = Word2Vec.load(mdlf)
        model.train(_corpus, total_examples=1, epochs=1)
    else:
        model = Word2Vec(
            sentences=_corpus,
            vector_size=256,
            window=1,
            min_count=1,
            workers=4,
        )
        model.save(mdlf)

    print(f"\n{cid:02d}", np.mean(model.wv[['29', '2', '3', '23']], axis=0)[:16])
    # break
