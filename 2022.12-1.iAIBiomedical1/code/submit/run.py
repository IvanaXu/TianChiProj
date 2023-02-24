#
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.mixture import GaussianMixture

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tcdata", type=str, default=".", help="tcdata")
parser.add_argument("--train_path", type=str, default=".", help="the path to get the train data")
parser.add_argument("--model_path", type=str, default=".", help="the path to save model")
args = parser.parse_args()
print(f">>> {args.tcdata}, {args.train_path}, {args.model_path}")

dA = pd.read_csv(f"{args.tcdata}/Affinity_test.csv", encoding="gbk")
print(dA.shape, dA.columns)

dN = pd.read_csv(f"{args.tcdata}/Neutralization_test.csv", encoding="gbk")
print(dN.shape, dN.columns)

_A = [True for i in range(len(dA))]
_N = [True for i in range(len(dN))]
_m = max([len(_A), len(_N)])
print(len(_A), len(_N), _m)

_A = (_A + [False] * _m)[:_m]
_N = (_N + [False] * _m)[:_m]
print(len(_A), len(_N), _m)

COL = [
    'Name', 
    'Alias', 
    'Ab or Nb', 
    'Protein+Epitope', 
    'Epitope class',
    'Binds to', 
    'Not Bind to', 
    'Neutralising Vs', 
    'Not Neutralising Vs',
    'SPR RBD (KD; nm)', 
    'SPR S1 (KD; nm)', 
    'SPR S2 (KD; nm)',
    'SPR S-ECD (KD; nm)', 
    'SPR S (KD; nm)', 
    'SPR NTD (KD; nm)',
    'SPR N (KD; nm)', 
    'BLI RBD (KD; nm)', 
    'BLI S1 (KD; nm)',
    'BLI S (KD; nm)', 
    'BLI NTD (KD; nm)', 
    'BLI N (KD; nm)',
    'MST RBD (KD; nm)', 
    'ELISA RBD competitive (IC50; μg/ml)',
    'ELISA S1 competitive (IC50; μg/ml)',
    'ELISA S competitive (IC50; μg/ml)',
    'ELISA S competitive (IC80; μg/ml)',
    'ELISA NTD competitive (IC50; μg/ml)',
    'ELISA RBD binding (EC50; μg/ml)', 
    'ELISA S1 binding (EC50; μg/ml)',
    'ELISA S binding (EC50; μg/ml)', 
    'ELISA N binding (EC50; μg/ml)',
    'FACS RBD (IC50; nm/ml)', 
    'FACS S (IC50; nm/ml)',
    'Live Virus Neutralisation IC50 (50% titre; μg/ml)阈值2μg/ml',
    'Live Virus Neutralisation IC80 (80% titre; μg/ml)',
    'Live Virus Neutralisation IC90 (90% titre; μg/ml)',
    'Live Virus Neutralisation IC100 (100% titre; μg/ml)',
    'Pseudo Virus Neutralisation IC50 (50% titre; μg/ml)',
    'Pseudo Virus Neutralisation IC80 (80% titre; μg/ml)',
    'Pseudo Virus Neutralisation IC90 (90% titre; μg/ml)',
    'Pseudo Virus Neutralisation IC100 (100% titre; μg/ml)',
    'Pseudo Virus Neutralisation (fold change)', 
    'Source', 
    'Source-update',
    'Notes', 
    'Origin', 
    'VH or VHH', 
    'VL', 
    'Heavy V Gene', 
    'Heavy J Gene',
    'Light V Gene', 
    'Light J Gene', 
    'CDRH3', 
    'CDRL3', 
    'PDB'
]

def feature(df):
    df["content"] = ""
    for icol in COL:
        df["content"] = df["content"] + " # " + df[icol].apply(str)
    return df

dA, dN = feature(dA), feature(dN)

#
dA_corpus = []
for i in dA["content"]:
    dA_corpus.append([j for j in i])
print(dA_corpus[:1])

dN_corpus = []
for i in dN["content"]:
    dN_corpus.append([j for j in i])
print(dN_corpus[:1])

_corpus = dA_corpus + dN_corpus
model = Word2Vec.load(f"{args.model_path}/model.mdl")
model.train(_corpus, total_examples=1, epochs=1)


#
def get_data(x):
    return pd.DataFrame(
        [
            np.concatenate([
                np.min(model.wv[i], axis=0),
                np.mean(model.wv[i], axis=0),
                np.median(model.wv[i], axis=0),
                np.max(model.wv[i], axis=0),
                np.std(model.wv[i], axis=0),
            ])
            for i in tqdm(x)
        ]
    )

dA_X = get_data(dA_corpus)
print(dA_X)

dN_X = get_data(dN_corpus)
print(dN_X)

modelA = GaussianMixture(n_components=5, covariance_type='full', random_state=10086)
modelA.fit(dA_X)
resultA = modelA.fit_predict(dA_X) + 1
print("A", resultA)

modelN = GaussianMixture(n_components=5, covariance_type='full', random_state=10086)
modelN.fit(dN_X)
resultN = modelN.fit_predict(dN_X) + 1
print("N", resultN)

#
answer = [1, 2, 3, 4, 5]
with open("result.csv", "w", encoding="gbk") as f:
    f.write("label_a,label_n\n")

    for m in tqdm(range(_m)):
        _rA = resultA[m] if _A[m] else ""
        _rN = resultN[m] if _N[m] else ""
        
        f.write(f"""{_rA},{_rN}\n""")

result = pd.read_csv("result.csv", sep=",")
print(result, result.shape, result.dtypes, result.describe())
print(pd.value_counts(result["label_a"], dropna=False, normalize=True), "\n")
print(pd.value_counts(result["label_n"], dropna=False, normalize=True), "\n")
print("END\n")
