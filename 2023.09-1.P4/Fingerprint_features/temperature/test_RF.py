"""
Fingerprint_features
使用随机森林回归模型
"""

# 导入所需的python库

###########Standard_Python_Libraries#######################
import os
import numpy as np
import random
from matplotlib import pyplot as plt

##############rdkit_library##########################
"""
http://www.rdkit.org/
RDKit是一个化学信息学和机器学习软件的集合库
使用rdkit来生成拓扑分子指纹
"""
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

#########panda to deal with csv files##############
import pandas as pd

###########sklearn_libary for ML models###################
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

#########yaml library to deal with "settings.yml"(containing various parameters) file#############
import yaml

##################################################
# Here we import another python code as a libray,
# this python codes simply read metal full electronic
# configuration and its oxidation state from the given csv file
# 该python文件从csv文件中读取不同原子轨道的电子占用和金属的氧化态
import encode_full_electronic_configuration

#############Print the Versions of the Sklearn and rdkit library#####################

print("   ###   Libraries:")
print("   ---   sklearn:{}".format(sklearn.__version__))
print("   ---   rdkit:{}".format(rdkit.__version__))
#########seed########################################
np.random.seed(1)
random.seed(1)


#######Defining the mean absolute error(mae) and r2 as an output of a function#########
"""
计算mae、r2、mape和smape指标函数
"""

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def SMAPE(y_true, y_pred):
    return (
        2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
    )


def reg_stats(y_true, y_pred, scaler=None):
    """
    Calculates the r^2 MAE between the true and predicted labels.

    Args:
        y_true (array-like): The true labels.
        y_pred (array-like): The predicted labels.
        scaler (sklearn.preprocessing.StandardScaler, optional): A scaler object to unscale the target values before calculating MAE. Defaults to None.
    Returns:
        tuple: A tuple containing the metrics of model.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if scaler:
        y_true_unscaled = scaler.inverse_transform(y_true.reshape(1, -1)).squeeze()
        y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(1, -1)).squeeze()
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    mape = MAPE(y_true_unscaled, y_pred_unscaled)
    smape = SMAPE(y_true_unscaled, y_pred_unscaled)
    return r2, mae, mape, smape


###########Training of Machine Learing Model given the output("target") and the various parmeters of the model###############
def train(df, target, hparam):
    """
    十折交叉训练和测试函数
    10-fold cross-train and test functions

    Args:
        df (pandas database): Dataset.
        target (string): Label that need to be predicted.
        hparam(yaml.load value): Parameters of yaml file.
    """

    fontname = "Arial"
    outdir = os.getcwd()

    print("start training")

    # 如果文件夹不存在，则创建需要的文件夹
    if not os.path.exists("%s/scatter_plots" % (outdir)):
        os.makedirs("%s/scatter_plots" % (outdir))

    if not os.path.exists("%s/models" % (outdir)):
        os.makedirs("%s/models" % (outdir))

    if not os.path.exists("%s/predictions_rawdata" % (outdir)):
        os.makedirs("%s/predictions_rawdata" % (outdir))

    # reading keywords from "settings.yml"
    # 读取配置文件"settings.yml"中的参数
    use_rdkit = hparam["use_rdkit"]
    rdkit_l = hparam["rdkit_l"]  # good number is 3-7
    rdkit_s = hparam["rdkit_s"]  # good number is 2**10 - 2**13
    # fraction_training=hparam["fraction_training"] # 0.8

    ##### dividing the full data in 10 different train and test set############
    # 将全部数据划分为10份，并使用十折交叉验证法进行训练和测试
    X = np.array(df.index.tolist())
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X)
    # sid=random.uniform(1,100000)
    counter = 0
    # now train ML model over all these 10 different train test split###########
    for train_index, test_index in kf.split(X):
        counter = counter + 1

        ###defining the output of the ML model
        # 数据归一化预处理
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(
            df[target].values.reshape(-1, 1)
        )  ###standard scaling of the output
        y_train, y_test = y[train_index], y[test_index]

        ##############preparing the input for the ML model########################################
        ##Reading the full electronic coufigurations of the metal. The electronic occupancy of the different
        # atomic orbitals are given as input.
        ##More details are provided in the "encode_electronic_configuration.py" code
        # 通过"encode_electronic_configuration.py"代码库读入不同原子轨道的电子占用作为输入特征
        e1 = encode_full_electronic_configuration.s1
        x_unscaled_feat_1 = e1
        x_feat_1 = x_unscaled_feat_1

        e2 = encode_full_electronic_configuration.s2
        x_unscaled_feat_2 = e2
        x_feat_2 = x_unscaled_feat_2

        e3 = encode_full_electronic_configuration.s3
        x_unscaled_feat_3 = e3
        x_feat_3 = x_unscaled_feat_3

        e4 = encode_full_electronic_configuration.s4
        x_unscaled_feat_4 = e4
        x_feat_4 = x_unscaled_feat_4

        e5 = encode_full_electronic_configuration.s5
        x_unscaled_feat_5 = e5
        x_feat_5 = x_unscaled_feat_5

        e6 = encode_full_electronic_configuration.s6
        x_unscaled_feat_6 = e6
        x_feat_6 = x_unscaled_feat_6

        e7 = encode_full_electronic_configuration.p2
        x_unscaled_feat_7 = e7
        x_feat_7 = x_unscaled_feat_7

        e8 = encode_full_electronic_configuration.p3
        x_unscaled_feat_8 = e8
        x_feat_8 = x_unscaled_feat_8

        e9 = encode_full_electronic_configuration.p4
        x_unscaled_feat_9 = e9
        x_feat_9 = x_unscaled_feat_9

        e10 = encode_full_electronic_configuration.p5
        x_unscaled_feat_10 = e10
        x_feat_10 = x_unscaled_feat_7

        e11 = encode_full_electronic_configuration.d3
        x_unscaled_feat_11 = e11
        x_feat_11 = x_unscaled_feat_11

        e12 = encode_full_electronic_configuration.d4
        x_unscaled_feat_12 = e12
        x_feat_12 = x_unscaled_feat_12

        e13 = encode_full_electronic_configuration.d5
        x_unscaled_feat_13 = e13
        x_feat_13 = x_unscaled_feat_13

        e14 = encode_full_electronic_configuration.f4
        x_unscaled_feat_14 = e14
        x_feat_14 = x_unscaled_feat_14

        # metal oxidation state are also read as input####
        # 金属氧化态同样作为输入
        e15 = encode_full_electronic_configuration.o
        x_unscaled_feat_15 = e15
        x_feat_15 = x_unscaled_feat_15

        ###rdkit fingerprint of the linkers (linker 1 and linker 2) are calculated now
        """
        通过csv文件中给出的linker，使用rdkit库的Chem模块获得fingerprint,
        将linker特征转化为数字向量特征
        """
        if use_rdkit:
            x_unscaled_fp1 = np.array(
                [
                    Chem.RDKFingerprint(mol1, maxPath=rdkit_l, fpSize=rdkit_s)
                    for mol1 in df["mol1"].tolist()
                ]
            ).astype(float)
            x_scaler_fp1 = StandardScaler()
            x_fp1 = x_scaler_fp1.fit_transform(x_unscaled_fp1)

            """
            x_unscaled_fp2 = np.array([Chem.RDKFingerprint(mol2, maxPath=rdkit_l, fpSize=rdkit_s) for mol2 in df['mol2'].tolist()]).astype(float)
            x_scaler_fp2 = StandardScaler()
            x_fp2 = x_scaler_fp2.fit_transform(x_unscaled_fp2)
            """

        ##Now combine all the features together to prepare the full input as x
        # 将所有特征组合在一起，获得完整的输入特征向量
        x = np.hstack(
            [
                x_fp1,
                x_feat_1,
                x_feat_2,
                x_feat_3,
                x_feat_4,
                x_feat_5,
                x_feat_6,
                x_feat_7,
                x_feat_8,
                x_feat_9,
                x_feat_10,
                x_feat_11,
                x_feat_12,
                x_feat_13,
                x_feat_14,
                x_feat_15,
            ]
        )
        x_unscaled = np.hstack(
            [
                x_unscaled_fp1,
                x_unscaled_feat_1,
                x_unscaled_feat_2,
                x_unscaled_feat_3,
                x_unscaled_feat_4,
                x_unscaled_feat_5,
                x_unscaled_feat_6,
                x_unscaled_feat_7,
                x_unscaled_feat_8,
                x_unscaled_feat_9,
                x_unscaled_feat_10,
                x_unscaled_feat_11,
                x_unscaled_feat_12,
                x_unscaled_feat_13,
                x_unscaled_feat_14,
                x_unscaled_feat_15,
            ]
        )

        # x = np.hstack([x_fp1,x_fp2,x_feat_1,x_feat_2,x_feat_3,x_feat_4,x_feat_5,x_feat_6,x_feat_7,x_feat_8,x_feat_9,x_feat_10,x_feat_11,x_feat_12,x_feat_13,x_feat_14,x_feat_15])
        # x_repeat = np.hstack([x_fp2,x_fp1,x_feat_1,x_feat_2,x_feat_3,x_feat_4,x_feat_5,x_feat_6,x_feat_7,x_feat_8,x_feat_9,x_feat_10,x_feat_11,x_feat_12,x_feat_13,x_feat_14,x_feat_15])
        # x_unscaled = np.hstack([x_unscaled_fp1,x_unscaled_fp2,x_unscaled_feat_1,x_unscaled_feat_2,x_unscaled_feat_3,x_unscaled_feat_4,x_unscaled_feat_5,x_unscaled_feat_6,x_unscaled_feat_7,x_unscaled_feat_8,x_unscaled_feat_9,x_unscaled_feat_10,x_unscaled_feat_11,x_unscaled_feat_12,x_unscaled_feat_13,x_unscaled_feat_14,x_unscaled_feat_15])
        # x_unscaled_repeat = np.hstack([x_unscaled_fp2,x_unscaled_fp1,x_unscaled_feat_1,x_unscaled_feat_2,x_unscaled_feat_3,x_unscaled_feat_4,x_unscaled_feat_5,x_unscaled_feat_6,x_unscaled_feat_7,x_unscaled_feat_8,x_unscaled_feat_9,x_unscaled_feat_10,x_unscaled_feat_11,x_unscaled_feat_12,x_unscaled_feat_13,x_unscaled_feat_14,x_unscaled_feat_15])

        ##separating the input in train and test set##
        x_train, x_test = x[train_index], x[test_index]
        x_unscaled_train, x_unscaled_test = (
            x_unscaled[train_index],
            x_unscaled[test_index],
        )

        """
        ####This part of the code checks whether there are two different linker present in a MOF.
        ###If two different linkers are present then two different data point are preapared for a single MOF 
        ##by concating the fingerprint of the linkers A and B(with fingerprint fp_A and fp_B ) 
        ##two different way as [fp_A, fp_B] and also [fp_B, fp_A]
        ##If there is only one type of linker present then only data point is generated as [fp_A, fp_A]
        
        # 如果MOF中有两个不同的linker，
        # 那么连接两个fingerprint的不同顺序[fp_A, fp_B]将[fp_B, fp_A]将视为两个不同的样本
        #Here we collect the index of the training data where there are two different linkers in the MOF
        r_train_index=[]
        for abcde in range(0,len(train_index)):
            if (df["nlinker"][train_index[abcde]])==2: 
                r_train_index.append(train_index[abcde])   

        #collects the index of the test data where there are two different  linkers in th MOF
        r_test_index=[]
        for abcde in range(0,len(test_index)):
            if (df["nlinker"][test_index[abcde]])==2:
                r_test_index.append(test_index[abcde])


        ### From the index collected above, now we construct additional ( input and output data) data for the ML model
        x_r_train, x_r_test = x_repeat[r_train_index],x_repeat[r_test_index] 
        x_r_unscaled_train, x_r_unscaled_test = x_unscaled_repeat[r_train_index], x_unscaled_repeat[r_test_index]
        y_r_train, y_r_test = y[r_train_index], y[r_test_index]

        #### Now all data are combined together to prepare the total set
        x_train=np.vstack([x_train,x_r_train])  
        y_train=np.vstack([y_train,y_r_train])
        x_test=np.vstack([x_test,x_r_test])
        y_test=np.vstack([y_test,y_r_test])
        """

        # final training and test data dimensions are printed here
        print("   ---   Training and test data dimensions:")
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


        ###############################################
        """
        创建随机森林回归模型，
        并使用训练集进行训练
        """
        ###############################################
        model = RandomForestRegressor(max_depth=7)
        # fit the model
        model.fit(x_train, y_train.ravel())

        ####Evaluation of the performance of the fitted model
        ####over training and test data set
        # 使用训练集，对模型预测结果进行评估
        print("\n   ###   RandomForestRegressor:")
        y_pred_train = model.predict(x_train)
        r2_GBR_train, mae_GBR_train, mape_train, smape_train = reg_stats(
            y_train, y_pred_train, y_scaler
        )
        print("   ---   Training (r2, MAE): %.3f %.3f" % (r2_GBR_train, mae_GBR_train))
        print("   ---   Training (mape, smape): %.3f %.3f" % (mape_train, smape_train))
        # 使用测试集，对模型预测结果进行评估
        y_pred_test = model.predict(x_test)
        r2_GBR_test, mae_GBR_test, mape_test, smape_test = reg_stats(
            y_test, y_pred_test, y_scaler
        )
        print("   ---   Testing (r2, MAE): %.3f %.3f" % (r2_GBR_test, mae_GBR_test))
        print("   ---   Testing (mape, smape): %.3f %.3f" % (mape_test, smape_test))

        ### Here we scale back the output
        # 对模型输出结果进行反变换，得到最终结果
        y_test_unscaled = y_scaler.inverse_transform(y_test.reshape(1, -1)).squeeze()
        y_train_unscaled = y_scaler.inverse_transform(y_train.reshape(1, -1)).squeeze()
        y_pred_test_unscaled = y_scaler.inverse_transform(
            y_pred_test.reshape(1, -1)
        ).squeeze()
        y_pred_train_unscaled = y_scaler.inverse_transform(
            y_pred_train.reshape(1, -1)
        ).squeeze()

        #### Saving and plotting of the predictions
        # 保存真实标签和模型预测结果
        np.savetxt("%s/predictions_rawdata/y_real_test.txt" % (outdir), y_test_unscaled)
        np.savetxt(
            "%s/predictions_rawdata/y_real_train.txt" % (outdir), y_train_unscaled
        )
        np.savetxt(
            "%s/predictions_rawdata/y_RFR_test.txt" % (outdir), y_pred_test_unscaled
        )
        np.savetxt(
            "%s/predictions_rawdata/y_RFR_train.txt" % (outdir), y_pred_train_unscaled
        )

        np.savetxt(
            "./predictions_rawdata/y_real_" + str(counter) + "_test.txt",
            y_test_unscaled,
        )
        np.savetxt(
            "./predictions_rawdata/y_real_" + str(counter) + "_train.txt",
            y_train_unscaled,
        )
        np.savetxt(
            "./predictions_rawdata/y_RFR_" + str(counter) + "_test.txt",
            y_pred_test_unscaled,
        )
        np.savetxt(
            "./predictions_rawdata/y_RFR_" + str(counter) + "_train.txt",
            y_pred_train_unscaled,
        )

        plt.figure()
        plt.scatter(
            y_pred_train_unscaled,
            y_train_unscaled,
            marker="o",
            c="C1",
            label="Training: r$^2$ = %.3f" % (r2_GBR_train),
        )
        plt.scatter(
            y_pred_test_unscaled,
            y_test_unscaled,
            marker="o",
            c="C2",
            label="Testing: r$^2$ = %.3f" % (r2_GBR_test),
        )
        plt.scatter(
            y_pred_train_unscaled,
            y_train_unscaled,
            marker="o",
            c="C1",
            label="Training: MAE = %.3f" % (mae_GBR_train),
        )
        plt.scatter(
            y_pred_test_unscaled,
            y_test_unscaled,
            marker="o",
            c="C2",
            label="Testing: MAE = %.3f" % (mae_GBR_test),
        )
        plt.plot(y_train_unscaled, y_train_unscaled)
        plt.title("RandomForestRegressor")
        plt.ylabel("Experimental Temperature [C]")
        plt.xlabel("Predicted Temperature [C]")
        plt.legend(loc="upper left")
        plt.savefig("%s/scatter_plots/full_data_RFR.png" % (outdir), dpi=300)
        plt.close()


# target: 模型预测内容
target = "temperature"  # Here output of the ML model is temperature

# 读取csv特征文件
df = pd.read_csv(
    "finger_example.csv"
)  ## The csv file containing the input output of the ML models

# 使用RDKit库中的Chem模块读取特征文件中的linker数据
# read the linker1 smiles and construct the molecule as mol1
df["mol1"] = df["linker1smi"].apply(lambda smi: Chem.MolFromSmiles(smi))

# read the linker2 smiles and construct the  molecule as mol2
#df['mol2']=df['linker2smi'].apply(lambda smi: Chem.MolFromSmiles(smi))


# Check and load the settings.yml file
# 读取配置文件settings.yml
if os.path.exists("settings.yml"):
    user_settings = yaml.load(open("settings.yml", "r"), Loader=yaml.SafeLoader)
    hparam = yaml.load(open("settings.yml", "r"), Loader=yaml.SafeLoader)

# 模型训练和测试
train(df, target, hparam)  # call the function defined above to train the ML model
