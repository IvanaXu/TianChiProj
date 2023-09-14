"""
RAC_features
使用神经网络回归模型
"""

###########Standard_Python_Libraries#######################
import os
import sys
import numpy as np
import random
import joblib
from matplotlib import pyplot as plt

##############rdkit_library##########################
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

##################### pytorch ###########################
import torch
import torch.nn as nn
import torch.optim as optim


#############Print the Versions of the Sklearn and rdkit library#####################

print("   ###   Libraries:")
print("   ---   sklearn:{}".format(sklearn.__version__))
print("   ---   rdkit:{}".format(rdkit.__version__))
print("   ---   Pytorch:{}".format(torch.__version__))


#############pytorch net##################
"""
This script defines three neural network models: LSTM, RNN, and CNNNet. 
The LSTM and RNN models are defined to take in input, output, midlayer1, and midlayer2 as parameters. 
The LSTM model uses a sequential LSTM layer followed by a fully connected layer with a LeakyReLU activation function and a dropout layer. 
The RNN model uses a sequential RNN layer followed by a fully connected layer with a LeakyReLU activation function and a dropout layer.
The CNNNet model is defined to take in input, output, midlayer1, and midlayer2 as parameters. 
It uses a fully connected layer with a LeakyReLU activation function and a dropout layer, 
    followed by a convolutional layer with a kernel size of 3 and a stride of 1, 
    and another fully connected layer with a LeakyReLU activation function and a dropout layer.
"""
# 1D CNN网络
class CNNNet(nn.Module):
    def __init__(self, input, output, midlayer1, midlayer2):
        """
        A convolutional neural network (CNN) model for regression.

        Args:
        input (int): The number of input features.
        output (int): The number of output features.
        midlayer1 (int): The number of neurons in the first hidden layer.
        midlayer2 (int): The number of neurons in the second hidden layer.

        Attributes:
        fulllinear (nn.Sequential): A fully connected neural network with three layers.
        full (nn.Sequential): A convolutional neural network with two fully connected layers.

        Methods:
        forward(x): Defines the forward pass of the CNN model.

        """
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
        """
        Defines the forward pass of the CNN model.

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor.

        """
        x = self.full(x)
        return x


# LSTM网络
class LSTM(nn.Module):
    def __init__(self, input, output, midlayer1, midlayer2):
        """
        A long short-term memory (LSTM) model for regression.

        Args:
        input (int): The number of input features.
        output (int): The number of output features.
        midlayer1 (int): The number of neurons in the first hidden layer.
        midlayer2 (int): The number of neurons in the second hidden layer.

        Attributes:
        lstm (nn.Sequential): A long short-term memory (LSTM) layer.
        full (nn.Sequential): A fully connected neural network with two layers.

        Methods:
        forward(x): Defines the forward pass of the LSTM model.

        """
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
        """
        Defines the forward pass of the LSTM model.

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor.

        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.full(out)
        return out


# RNN网络
class RNN(nn.Module):
    def __init__(self, input, output, midlayer1, midlayer2):
        super(RNN, self).__init__()
        """
        A recurrent neural network (RNN) model for regression.

        Args:
        input (int): The number of input features.
        output (int): The number of output features.
        midlayer1 (int): The number of neurons in the first hidden layer.
        midlayer2 (int): The number of neurons in the second hidden layer.

        Attributes:
        rnn (nn.Sequential): A recurrent neural network (RNN) layer.
        full (nn.Sequential): A fully connected neural network with two layers.

        Methods:
        forward(x): Defines the forward pass of the RNN model.

        """
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
        """
        Defines the forward pass of the RNN model.

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor.

        """
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.full(out)
        return out


#######Defining the mean absolute error(mae) and r2 as an output of a function#########
def reg_stats(y_true, y_pred, scaler=None):
    """
    计算mae和r2指标函数
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
        y_true_unscaled = scaler.inverse_transform(y_true)
        y_pred_unscaled = scaler.inverse_transform(y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    return r2, mae


###########Training of ML Model given the input("df") and the output("target")of the model###############
def train(df, target):
    """
    十折交叉训练和测试函数
    10-fold cross-train and test functions

    Args:
        df (pandas-database): Dataset.
        target (string): Label that need to be predicted.
    """

    fontname = "Arial"
    outdir = os.getcwd()

    print("start training")

    if not os.path.exists("%s/scatter_plots" % (outdir)):
        os.makedirs("%s/scatter_plots" % (outdir))

    if not os.path.exists("%s/models" % (outdir)):
        os.makedirs("%s/models" % (outdir))

    if not os.path.exists("%s/predictions_rawdata" % (outdir)):
        os.makedirs("%s/predictions_rawdata" % (outdir))

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
        # 对样本标签进行数据归一化预处理
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(df[target].values.reshape(-1, 1))
        y_train, y_test = y[train_index], y[test_index]

        ###Preparing the input according to precomputed features by KULIK
        ### and co-workers. All the features are loaded within features_basic array
        # 根据csv文件中的特征名选择需要作为输入的特征
        features_basic = [
            "ASA [m^2/cm^3]",
            "CellV [A^3]",
            "Df",
            "Di",
            "Dif",
            "NASA [m^2/cm^3]",
            "POAV [cm^3/g]",
            "POAVF",
            "PONAV [cm^3/g]",
            "PONAVF",
            "density [g/cm^3]",
            "total_SA_volumetric",
            "total_SA_gravimetric",
            "total_POV_volumetric",
            "total_POV_gravimetric",
            "mc_CRY-chi-0-all",
            "mc_CRY-chi-1-all",
            "mc_CRY-chi-2-all",
            "mc_CRY-chi-3-all",
            "mc_CRY-Z-0-all",
            "mc_CRY-Z-1-all",
            "mc_CRY-Z-2-all",
            "mc_CRY-Z-3-all",
            "mc_CRY-I-0-all",
            "mc_CRY-I-1-all",
            "mc_CRY-I-2-all",
            "mc_CRY-I-3-all",
            "mc_CRY-T-0-all",
            "mc_CRY-T-1-all",
            "mc_CRY-T-2-all",
            "mc_CRY-T-3-all",
            "mc_CRY-S-0-all",
            "mc_CRY-S-1-all",
            "mc_CRY-S-2-all",
            "mc_CRY-S-3-all",
            "D_mc_CRY-chi-0-all",
            "D_mc_CRY-chi-1-all",
            "D_mc_CRY-chi-2-all",
            "D_mc_CRY-chi-3-all",
            "D_mc_CRY-Z-0-all",
            "D_mc_CRY-Z-1-all",
            "D_mc_CRY-Z-2-all",
            "D_mc_CRY-Z-3-all",
            "D_mc_CRY-I-0-all",
            "D_mc_CRY-I-1-all",
            "D_mc_CRY-I-2-all",
            "D_mc_CRY-I-3-all",
            "D_mc_CRY-T-0-all",
            "D_mc_CRY-T-1-all",
            "D_mc_CRY-T-2-all",
            "D_mc_CRY-T-3-all",
            "D_mc_CRY-S-0-all",
            "D_mc_CRY-S-1-all",
            "D_mc_CRY-S-2-all",
            "D_mc_CRY-S-3-all",
            "sum-mc_CRY-chi-0-all",
            "sum-mc_CRY-chi-1-all",
            "sum-mc_CRY-chi-2-all",
            "sum-mc_CRY-chi-3-all",
            "sum-mc_CRY-Z-0-all",
            "sum-mc_CRY-Z-1-all",
            "sum-mc_CRY-Z-2-all",
            "sum-mc_CRY-Z-3-all",
            "sum-mc_CRY-I-0-all",
            "sum-mc_CRY-I-1-all",
            "sum-mc_CRY-I-2-all",
            "sum-mc_CRY-I-3-all",
            "sum-mc_CRY-T-0-all",
            "sum-mc_CRY-T-1-all",
            "sum-mc_CRY-T-2-all",
            "sum-mc_CRY-T-3-all",
            "sum-mc_CRY-S-0-all",
            "sum-mc_CRY-S-1-all",
            "sum-mc_CRY-S-2-all",
            "sum-mc_CRY-S-3-all",
            "sum-D_mc_CRY-chi-0-all",
            "sum-D_mc_CRY-chi-1-all",
            "sum-D_mc_CRY-chi-2-all",
            "sum-D_mc_CRY-chi-3-all",
            "sum-D_mc_CRY-Z-0-all",
            "sum-D_mc_CRY-Z-1-all",
            "sum-D_mc_CRY-Z-2-all",
            "sum-D_mc_CRY-Z-3-all",
            "sum-D_mc_CRY-I-0-all",
            "sum-D_mc_CRY-I-1-all",
            "sum-D_mc_CRY-I-2-all",
            "sum-D_mc_CRY-I-3-all",
            "sum-D_mc_CRY-T-0-all",
            "sum-D_mc_CRY-T-1-all",
            "sum-D_mc_CRY-T-2-all",
            "sum-D_mc_CRY-T-3-all",
            "sum-D_mc_CRY-S-0-all",
            "sum-D_mc_CRY-S-1-all",
            "sum-D_mc_CRY-S-2-all",
            "sum-D_mc_CRY-S-3-all",
            "lc-chi-0-all",
            "lc-chi-1-all",
            "lc-chi-2-all",
            "lc-chi-3-all",
            "lc-Z-0-all",
            "lc-Z-1-all",
            "lc-Z-2-all",
            "lc-Z-3-all",
            "lc-I-0-all",
            "lc-I-1-all",
            "lc-I-2-all",
            "lc-I-3-all",
            "lc-T-0-all",
            "lc-T-1-all",
            "lc-T-2-all",
            "lc-T-3-all",
            "lc-S-0-all",
            "lc-S-1-all",
            "lc-S-2-all",
            "lc-S-3-all",
            "lc-alpha-0-all",
            "lc-alpha-1-all",
            "lc-alpha-2-all",
            "lc-alpha-3-all",
            "D_lc-chi-0-all",
            "D_lc-chi-1-all",
            "D_lc-chi-2-all",
            "D_lc-chi-3-all",
            "D_lc-Z-0-all",
            "D_lc-Z-1-all",
            "D_lc-Z-2-all",
            "D_lc-Z-3-all",
            "D_lc-I-0-all",
            "D_lc-I-1-all",
            "D_lc-I-2-all",
            "D_lc-I-3-all",
            "D_lc-T-0-all",
            "D_lc-T-1-all",
            "D_lc-T-2-all",
            "D_lc-T-3-all",
            "D_lc-S-0-all",
            "D_lc-S-1-all",
            "D_lc-S-2-all",
            "D_lc-S-3-all",
            "D_lc-alpha-0-all",
            "D_lc-alpha-1-all",
            "D_lc-alpha-2-all",
            "D_lc-alpha-3-all",
            "func-chi-0-all",
            "func-chi-1-all",
            "func-chi-2-all",
            "func-chi-3-all",
            "func-Z-0-all",
            "func-Z-1-all",
            "func-Z-2-all",
            "func-Z-3-all",
            "func-I-0-all",
            "func-I-1-all",
            "func-I-2-all",
            "func-I-3-all",
            "func-T-0-all",
            "func-T-1-all",
            "func-T-2-all",
            "func-T-3-all",
            "func-S-0-all",
            "func-S-1-all",
            "func-S-2-all",
            "func-S-3-all",
            "func-alpha-0-all",
            "func-alpha-1-all",
            "func-alpha-2-all",
            "func-alpha-3-all",
            "D_func-chi-0-all",
            "D_func-chi-1-all",
            "D_func-chi-2-all",
            "D_func-chi-3-all",
            "D_func-Z-0-all",
            "D_func-Z-1-all",
            "D_func-Z-2-all",
            "D_func-Z-3-all",
            "D_func-I-0-all",
            "D_func-I-1-all",
            "D_func-I-2-all",
            "D_func-I-3-all",
            "D_func-T-0-all",
            "D_func-T-1-all",
            "D_func-T-2-all",
            "D_func-T-3-all",
            "D_func-S-0-all",
            "D_func-S-1-all",
            "D_func-S-2-all",
            "D_func-S-3-all",
            "D_func-alpha-0-all",
            "D_func-alpha-1-all",
            "D_func-alpha-2-all",
            "D_func-alpha-3-all",
            "f-lig-chi-0",
            "f-lig-chi-1",
            "f-lig-chi-2",
            "f-lig-chi-3",
            "f-lig-Z-0",
            "f-lig-Z-1",
            "f-lig-Z-2",
            "f-lig-Z-3",
            "f-lig-I-0",
            "f-lig-I-1",
            "f-lig-I-2",
            "f-lig-I-3",
            "f-lig-T-0",
            "f-lig-T-1",
            "f-lig-T-2",
            "f-lig-T-3",
            "f-lig-S-0",
            "f-lig-S-1",
            "f-lig-S-2",
            "f-lig-S-3",
            "sum-lc-chi-0-all",
            "sum-lc-chi-1-all",
            "sum-lc-chi-2-all",
            "sum-lc-chi-3-all",
            "sum-lc-Z-0-all",
            "sum-lc-Z-1-all",
            "sum-lc-Z-2-all",
            "sum-lc-Z-3-all",
            "sum-lc-I-0-all",
            "sum-lc-I-1-all",
            "sum-lc-I-2-all",
            "sum-lc-I-3-all",
            "sum-lc-T-0-all",
            "sum-lc-T-1-all",
            "sum-lc-T-2-all",
            "sum-lc-T-3-all",
            "sum-lc-S-0-all",
            "sum-lc-S-1-all",
            "sum-lc-S-2-all",
            "sum-lc-S-3-all",
            "sum-lc-alpha-0-all",
            "sum-lc-alpha-1-all",
            "sum-lc-alpha-2-all",
            "sum-lc-alpha-3-all",
            "sum-D_lc-chi-0-all",
            "sum-D_lc-chi-1-all",
            "sum-D_lc-chi-2-all",
            "sum-D_lc-chi-3-all",
            "sum-D_lc-Z-0-all",
            "sum-D_lc-Z-1-all",
            "sum-D_lc-Z-2-all",
            "sum-D_lc-Z-3-all",
            "sum-D_lc-I-0-all",
            "sum-D_lc-I-1-all",
            "sum-D_lc-I-2-all",
            "sum-D_lc-I-3-all",
            "sum-D_lc-T-0-all",
            "sum-D_lc-T-1-all",
            "sum-D_lc-T-2-all",
            "sum-D_lc-T-3-all",
            "sum-D_lc-S-0-all",
            "sum-D_lc-S-1-all",
            "sum-D_lc-S-2-all",
            "sum-D_lc-S-3-all",
            "sum-D_lc-alpha-0-all",
            "sum-D_lc-alpha-1-all",
            "sum-D_lc-alpha-2-all",
            "sum-D_lc-alpha-3-all",
            "sum-func-chi-0-all",
            "sum-func-chi-1-all",
            "sum-func-chi-2-all",
            "sum-func-chi-3-all",
            "sum-func-Z-0-all",
            "sum-func-Z-1-all",
            "sum-func-Z-2-all",
            "sum-func-Z-3-all",
            "sum-func-I-0-all",
            "sum-func-I-1-all",
            "sum-func-I-2-all",
            "sum-func-I-3-all",
            "sum-func-T-0-all",
            "sum-func-T-1-all",
            "sum-func-T-2-all",
            "sum-func-T-3-all",
            "sum-func-S-0-all",
            "sum-func-S-1-all",
            "sum-func-S-2-all",
            "sum-func-S-3-all",
            "sum-func-alpha-0-all",
            "sum-func-alpha-1-all",
            "sum-func-alpha-2-all",
            "sum-func-alpha-3-all",
            "sum-D_func-chi-0-all",
            "sum-D_func-chi-1-all",
            "sum-D_func-chi-2-all",
            "sum-D_func-chi-3-all",
            "sum-D_func-Z-0-all",
            "sum-D_func-Z-1-all",
            "sum-D_func-Z-2-all",
            "sum-D_func-Z-3-all",
            "sum-D_func-I-0-all",
            "sum-D_func-I-1-all",
            "sum-D_func-I-2-all",
            "sum-D_func-I-3-all",
            "sum-D_func-T-0-all",
            "sum-D_func-T-1-all",
            "sum-D_func-T-2-all",
            "sum-D_func-T-3-all",
            "sum-D_func-S-0-all",
            "sum-D_func-S-1-all",
            "sum-D_func-S-2-all",
            "sum-D_func-S-3-all",
            "sum-D_func-alpha-0-all",
            "sum-D_func-alpha-1-all",
            "sum-D_func-alpha-2-all",
            "sum-D_func-alpha-3-all",
            "sum-f-lig-chi-0",
            "sum-f-lig-chi-1",
            "sum-f-lig-chi-2",
            "sum-f-lig-chi-3",
            "sum-f-lig-Z-0",
            "sum-f-lig-Z-1",
            "sum-f-lig-Z-2",
            "sum-f-lig-Z-3",
            "sum-f-lig-I-0",
            "sum-f-lig-I-1",
            "sum-f-lig-I-2",
            "sum-f-lig-I-3",
            "sum-f-lig-T-0",
            "sum-f-lig-T-1",
            "sum-f-lig-T-2",
            "sum-f-lig-T-3",
            "sum-f-lig-S-0",
            "sum-f-lig-S-1",
            "sum-f-lig-S-2",
            "sum-f-lig-S-3",
        ]

        ###standard scaling of the input features
        # 对样本特征进行数据归一化预处理
        x_scaler_feat = StandardScaler()
        x_unscaled_feat = df[features_basic].values
        x_feat = x_scaler_feat.fit_transform(x_unscaled_feat)
        n_feat = len(features_basic)
        x = x_feat
        x_unscaled = x_unscaled_feat

        ## Dividing the input in train and test data set
        x_train, x_test = x[train_index], x[test_index]
        x_unscaled_train, x_unscaled_test = (
            x_unscaled[train_index],
            x_unscaled[test_index],
        )

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
        # net =  LSTM(x_train.shape[1], 1, 512, 128).cuda()
        net = RNN(x_train.shape[1], 1, 512, 128).cuda()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        for epoch in range(100):
            running_loss = 0.0
            optimizer.zero_grad()

            net.train()
            # y_pred_train = net(torch.tensor(x_train).float())  # CNN
            y_pred_train = net(
                torch.tensor(x_train).float().unsqueeze(1).cuda()
            )  # LSTM
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

        # 绘制模型预测结果图像
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

        plt.ylabel("Experimental ")
        plt.xlabel("Predicted ")
        plt.legend(loc="upper left")
        plt.savefig("%s/scatter_plots/full_data_RFR.png" % (outdir), dpi=300)
        plt.close()


# target: 模型预测内容
target = "time"  # Here output of the ML model in time

# 读取csv特征文件
df = pd.read_csv(
    "example.csv"
)  ## The csv file containing the input output of the ML models

# 模型训练和测试
train(df, target)  # call the function defined above to train the ML model
