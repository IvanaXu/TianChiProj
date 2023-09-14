"""
Fingerprint_features
使用神经网络回归模型
"""

# First, We import the python libraries necesary to run this calculation

###########Standard_Python_Libraries#######################
import os
import numpy as np
import random
from matplotlib import pyplot as plt

################rdkit_library##########################
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

##################### pytorch ###########################
import torch
import torch.nn as nn
import torch.optim as optim


#########yaml library to deal with "settings.yml"(containing various parameters) file#############
import yaml

##################################################
# Here we import another python code as a libray,
# this python codes simply read metal full electronic
# configuration and its oxidation state from the given csv file
import encode_full_electronic_configuration

#############Print the Versions of the Sklearn and rdkit library#####################

print("   ###   Libraries:")
print("   ---   sklearn:{}".format(sklearn.__version__))
print("   ---   rdkit:{}".format(rdkit.__version__))

#########seed########################################
np.random.seed(1)
random.seed(1)


#############pytorch net##################
# 1D CNN网络
class CNNNet(nn.Module):
    def __init__(self, input, output, midlayer1, midlayer2):
        super(CNNNet, self).__init__()
        self.fulllinear = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(in_features=input, out_features=midlayer1, bias=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(0.2),
            nn.Linear(in_features=midlayer1, out_features=midlayer2, bias=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(0.2),
            nn.Linear(in_features=midlayer2, out_features=output, bias=True),
            nn.Sigmoid(),
        )
        self.full = nn.Sequential(
            # nn.Flatten(),
            nn.Conv1d(
                in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=input, out_features=midlayer1, bias=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(0.2),
            nn.Linear(in_features=midlayer1, out_features=output, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.full(x)
        return x


# LSTM网络
class LSTM(nn.Module):
    def __init__(self, input, output, midlayer1, midlayer2):
        super(LSTM, self).__init__()
        # self.lstm = nn.LSTM(input, midlayer1)
        self.lstm = nn.Sequential(
            # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.LSTM(input, midlayer1, 2, batch_first=True)
        )
        self.full = nn.Sequential(
            nn.Linear(in_features=midlayer1, out_features=midlayer2, bias=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(0.2),
            nn.Linear(in_features=midlayer2, out_features=output, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.full(out)
        return out


# RNN网络
class RNN(nn.Module):
    def __init__(self, input, output, midlayer1, midlayer2):
        super(RNN, self).__init__()
        self.rnn = nn.Sequential(
            # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.RNN(input, midlayer1, batch_first=True)
        )
        self.full = nn.Sequential(
            nn.Linear(in_features=midlayer1, out_features=midlayer2, bias=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(0.2),
            nn.Linear(in_features=midlayer2, out_features=output, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.full(out)
        return out


#######Defining the mean absolute error(mae) and r2 as an output of a function#########
"""
计算mae和r2指标函数
输入真实标签和预测结果
"""

def reg_stats(y_true, y_pred, scaler=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if scaler:
        y_true_unscaled = scaler.inverse_transform(y_true.reshape(1, -1)).squeeze()
        y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(1, -1)).squeeze()
    else:
        y_true_unscaled = y_true
        y_pred_unscaled = y_pred
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    return r2, mae


###########Training of Machine Learing Model given the output("target") and the various parmeters of the model###############
"""
十折交叉训练和测试函数
"""

def train(df, target, hparam):
    fontname = "Arial"
    outdir = os.getcwd()

    print("start training")

    if not os.path.exists("%s/scatter_plots" % (outdir)):
        os.makedirs("%s/scatter_plots" % (outdir))

    if not os.path.exists("%s/models" % (outdir)):
        os.makedirs("%s/models" % (outdir))

    if not os.path.exists("%s/predictions_rawdata" % (outdir)):
        os.makedirs("%s/predictions_rawdata" % (outdir))

    # reading keywords from "settings.yml"

    use_rdkit = hparam["use_rdkit"]
    rdkit_l = hparam["rdkit_l"]  # good number is 3-7
    rdkit_s = hparam["rdkit_s"]  # good number is 2**10 - 2**13
    # fraction_training=hparam["fraction_training"] # 0.8

    ##### dividing the full data in 10 different train and test set############
    X = np.array(df.index.tolist())
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X)
    # sid=random.uniform(1,100000)
    counter = 0
    # now train ML model over all these 10 different train test split###########
    for train_index, test_index in kf.split(X):
        counter = counter + 1

        ###defining the output of the ML model
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
        # 读取金属氧化态同样作为输入
        e15 = encode_full_electronic_configuration.o
        x_unscaled_feat_15 = e15
        x_feat_15 = x_unscaled_feat_15

        ###rdkit fingerprint of the linkers (linker 1 and linker 2) are calculated now
        """
        通过csv文件中给出的linker1和linker2，使用rdkit库得到fingerprint
        即将linker1和linker2特征转化为数字向量特征
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

            x_unscaled_fp2 = np.array(
                [
                    Chem.RDKFingerprint(mol2, maxPath=rdkit_l, fpSize=rdkit_s)
                    for mol2 in df["mol2"].tolist()
                ]
            ).astype(float)
            x_scaler_fp2 = StandardScaler()
            x_fp2 = x_scaler_fp2.fit_transform(x_unscaled_fp2)

        ##Now combine all the features together to prepare the full input as x
        # 将所有特征组合在一起，获得完整的输入特征向量
        x = np.hstack(
            [
                x_fp1,
                x_fp2,
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
        x_repeat = np.hstack(
            [
                x_fp2,
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
                x_unscaled_fp2,
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
        x_unscaled_repeat = np.hstack(
            [
                x_unscaled_fp2,
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

        ##separating the input in train and test set#
        # 将所有特征组合在一起，获得完整的特征输入
        x_train, x_test = x[train_index], x[test_index]
        x_unscaled_train, x_unscaled_test = (
            x_unscaled[train_index],
            x_unscaled[test_index],
        )

        ####This part of the code checks whether there are two different linker present in a MOF.
        ###If two different linkers are present then two different data point are preapared for a single MOF
        ##by concating the fingerprint of the linkers A and B(with fingerprint fp_A and fp_B )
        ##two different way as [fp_A, fp_B] and also [fp_B, fp_A]
        ##If there is only one type of linker present then only data point is generated as [fp_A, fp_A]
        """
        如果MOF中有两个不同的linker，
        那么连接两个fingerprint的不同顺序[fp_A, fp_B]将[fp_B, fp_A]将视为两个不同的样本
        """
        # Here we collect the index of the training data where there are two different linkers in the MOF

        r_train_index = []
        for abcde in range(0, len(train_index)):
            if (df["nlinker"][train_index[abcde]]) == 2:
                r_train_index.append(train_index[abcde])

        # collects the index of the test data where there are two different  linkers in th MOF
        r_test_index = []
        for abcde in range(0, len(test_index)):
            if (df["nlinker"][test_index[abcde]]) == 2:
                r_test_index.append(test_index[abcde])

        ### From the index collected above, now we construct additional ( input and output data) data for the ML models

        x_r_train, x_r_test = x_repeat[r_train_index], x_repeat[r_test_index]
        x_r_unscaled_train, x_r_unscaled_test = (
            x_unscaled_repeat[r_train_index],
            x_unscaled_repeat[r_test_index],
        )
        y_r_train, y_r_test = y[r_train_index], y[r_test_index]

        #### Now all data are combined together to prepare the total set

        x_train = np.vstack([x_train, x_r_train])
        y_train = np.vstack([y_train, y_r_train])
        x_test = np.vstack([x_test, x_r_test])
        y_test = np.vstack([y_test, y_r_test])

        # final training and test data dimensions are printed here

        print("\n   ---   Training and test data dimensions:")
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        ###############################################
        """
        创建神经网络回归模型，
        并使用训练集进行训练
        """
        ###############################################
        # 定义模型、损失函数和优化器
        net = LSTM(x_train.shape[1], 1, 512, 128).cuda()
        # net =  RNN(x_train.shape[1], 1, 128, 64).cuda()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        for epoch in range(100):
            running_loss = 0.0
            optimizer.zero_grad()

            net.train()
            # y_pred_train = net(torch.tensor(x_train).float())  # CNN
            y_pred_train = net(
                torch.tensor(x_train).float().unsqueeze(1).cuda()
            )  # LSTM RNN
            loss = criterion(y_pred_train, torch.tensor(y_train).float().cuda())  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item()
            """
            print('epoch %d，MSE loss：%.3f' % (epoch+1, running_loss))

            r2_GBR_train,mae_GBR_train = reg_stats(y_train, y_pred_train.detach().numpy(), y_scaler)
            print("   ---   Training (r2, MAE): %.3f %.3f"%(r2_GBR_train,mae_GBR_train))

            net.eval()
            y_pred_test = net(torch.tensor(x_test).float())
            r2_GBR_test,mae_GBR_test = reg_stats(y_test, y_pred_test.detach().numpy(), y_scaler)
            print("   ---   Testing (r2, MAE): %.3f %.3f"%(r2_GBR_test,mae_GBR_test))
            """

        # 使用训练集和测试集，对模型预测结果进行评估
        net.eval()
        r2_GBR_train, mae_GBR_train = reg_stats(
            y_train, y_pred_train.detach().cpu().numpy(), y_scaler
        )
        print("   ---   Training (r2, MAE): %.3f %.3f" % (r2_GBR_train, mae_GBR_train))
        y_pred_test = net(torch.tensor(x_test).float().unsqueeze(1).cuda())
        r2_GBR_test, mae_GBR_test = reg_stats(
            y_test, y_pred_test.detach().cpu().numpy(), y_scaler
        )
        print("   ---   Testing (r2, MAE): %.3f %.3f" % (r2_GBR_test, mae_GBR_test))

        ### Here we scale back the output
        # 对模型输出结果进行反变换，得到最终结果
        y_test_unscaled = y_scaler.inverse_transform(y_test.reshape(1, -1)).squeeze()
        y_train_unscaled = y_scaler.inverse_transform(y_train.reshape(1, -1)).squeeze()
        y_pred_test_unscaled = y_scaler.inverse_transform(
            y_pred_test.detach().cpu().numpy().reshape(1, -1)
        ).squeeze()
        y_pred_train_unscaled = y_scaler.inverse_transform(
            y_pred_train.detach().cpu().numpy().reshape(1, -1)
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
        plt.title("LSTM")
        plt.ylabel("Experimental")
        plt.xlabel("Predicted")
        plt.legend(loc="upper left")
        plt.savefig("%s/scatter_plots/full_data_RFR.png" % (outdir), dpi=300)
        plt.close()


# target: 模型预测内容
target = "time"  # Here output of the ML model in time

# 读取csv特征文件
df = pd.read_csv(
    "example.csv"
)  ## The csv file containing the input output of the ML models

# 使用RDKit库中的Chem模块读取特征文件中的linker数据
# read the linker1 smiles and construct the molecule as mol1
df["mol1"] = df["linker1smi"].apply(lambda smi: Chem.MolFromSmiles(smi))

# read the linker2 smiles and construct the  molecule as mol2
df["mol2"] = df["linker2smi"].apply(lambda smi: Chem.MolFromSmiles(smi))


# Check and load the settings.yml file
# 读取配置文件settings.yml
if os.path.exists("settings.yml"):
    user_settings = yaml.load(open("settings.yml", "r"), Loader=yaml.SafeLoader)
    hparam = yaml.load(open("settings.yml", "r"), Loader=yaml.SafeLoader)

# 模型训练和测试
train(df, target, hparam)  # call the function defined above to train the ML model
