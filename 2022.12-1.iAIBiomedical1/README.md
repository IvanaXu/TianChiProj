
### “云上进化”2022全球AI生物智药大赛：赛道一“基于AI算法的SARS-CoV-2广谱中和抗体药物设计”
### 一、团队信息
* 团队名：paipai
* score：0.24817
* 最优成绩提交日：2022-10-20

### 二、解决方案
> 1、未使用baseline

> 2、算法思想（文本处理 + 无监督学习）

对 Affinity_train、Neutralization_train，全字段进行文本拼接，使用“ # ”分隔
```python
def feature(df):
    df["content"] = ""
    for icol in COL:
        df["content"] = df["content"] + " # " + df[icol].apply(str)
    return df
```
如上函数，打造content字段，进一步组成_corpus，

使用文本模型进行训练并保存，
```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=_corpus,
    vector_size=256,
    window=1,
    min_count=1,
    seed=10086,
    workers=1,
)
model.save(f"{args.model_path}/model.mdl")
```
再进一步对向量化后特征进行256维min、mean等并concatenate 作为特征集$X_n$
```python
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
```
在特征集$X_n$的基础上，对Affinity、Neutralization分别建模，

> TODO 这里，与预测目标相关的数据清洗过程中，我们发现一些数据问题。

故而，先采用了无监督模型，将预测集拆解为5组并输出结果：
```python
from sklearn.mixture import GaussianMixture
model = GaussianMixture(
    n_components=5, covariance_type='full', random_state=10086)
```

> 3、展望

（1）TODO部分将在下一阶段进行；

（2）改进无监督模型 或预测目标引导的无监督模型 或有监督模型；

### 三、文件结构
```
project
├── README.md
├── code
│   ├── local.sh                        //本地运行脚本
│   ├── submit                             
│   │   ├── Dockerfile
│   │   ├── requirements.txt            //requirements python
│   │   ├── result.md5                  //测试tcdata输出结果一致
│   │   ├── run.py                      //预测脚本
│   │   ├── run.sh                        
│   │   ├── train.py                    //训练脚本
│   │   └── train_data
│   │       ├── Affinity_train.csv
│   │       └── Neutralization_train.csv
│   └── submit.sh                       //提交docker脚本，需修改basep
├── prediction_result
│   └── result.csv
├── tcdata                              //tcdata 根据train文件模拟，用于测试代码
│   ├── Affinity_test.csv
│   └── Neutralization_test.csv
└── user_data
    ├── model_data
    │   └── model.mdl                   //model文件
    └── train_data
        ├── Affinity_train.csv
        └── Neutralization_train.csv
```
程序入口（本地运行）：
```shell
cd project/code
sh local.sh
```
程序入口（提交docker）：
```shell
cd project/code
sh submit.sh
```
