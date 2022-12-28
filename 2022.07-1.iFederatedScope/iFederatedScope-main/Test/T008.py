import torch
import numpy as np
datp = "/Users/ivan/Desktop/Data/CIKM2022/CIKM22Competition/"
_data = torch.load(f"{datp}/11/train.pt")

top = 48

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = []
for idata in _data:
    corpus.append(
        " ".join([str(i) for i in np.array(idata.edge_index[1]) if i in range(top+1)])
    )
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.95)
tfidf.fit(corpus)

import gzip
import pickle
with gzip.GzipFile(f"{datp}/11.mdl", "wb") as f:
    pickle.dump(tfidf, f)

with gzip.GzipFile(f"{datp}/11.mdl", "rb") as f:
    mtfidf = pickle.load(f)

X = np.array(mtfidf.transform([corpus[0]]).todense())[0]
print(X.shape)
print(X[4])
