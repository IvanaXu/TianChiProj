"""
RAC_features
使用随机森林回归模型
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

#############Print the Versions of the Sklearn and rdkit library#####################

print("   ###   Libraries:")
print("   ---   sklearn:{}".format(sklearn.__version__))
print("   ---   rdkit:{}".format(rdkit.__version__))


#######Defining the mean absolute error(mae) and r2 as an output of a function#########
"""
计算mae和r2指标函数
输入真实标签和预测结果
"""


def reg_stats(y_true, y_pred, scaler=None):
    """
    Calculates the R2 and mean absolute error (MAE) metrics for regression models.

    Parameters:
    y_true (array-like): The true target values.
    y_pred (array-like): The predicted target values.
    scaler (object, optional): A scaler object to unscale the target values before calculating MAE. Defaults to None.

    Returns:
    tuple: A tuple containing the R2 and MAE metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if scaler:
        y_true_unscaled = scaler.inverse_transform(y_true.reshape(1, -1)).squeeze()
        y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(1, -1)).squeeze()
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    return r2, mae


###########Training of ML Model given the input(df) and  output("target") of the model###############
"""
十折交叉训练和测试函数
"""


def train(df, target):
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
        y = y_scaler.fit_transform(
            df[target].values.reshape(-1, 1)
        )  # standard scaling of the output
        ## Diving the output in train and test data
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
        print("   ---   Training and test data dimensions:")
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        ###############################################
        """
        创建随机森林回归模型，
        并使用训练集进行训练
        """
        ###############################################
        model = RandomForestRegressor(max_depth=5)

        # fit the model
        model.fit(x_train, y_train.ravel())

        ####Evaluation of the performance of the fitted model
        ####over training and test data set
        # 使用训练集，对模型预测结果进行评估
        print("\n   ###   RandomForestRegressor:")
        y_pred_train = model.predict(x_train)
        r2_GBR_train, mae_GBR_train = reg_stats(y_train, y_pred_train, y_scaler)
        print("   ---   Training (r2, MAE): %.3f %.3f" % (r2_GBR_train, mae_GBR_train))
        # 使用测试集，对模型预测结果进行评估
        y_pred_test = model.predict(x_test)
        r2_GBR_test, mae_GBR_test = reg_stats(y_test, y_pred_test, y_scaler)
        print("   ---   Testing (r2, MAE): %.3f %.3f" % (r2_GBR_test, mae_GBR_test))
        # joblib.dump(model, "%s/models/random_forest_regression.joblib"%(outdir))

        ### Here we scale back the output
        # 对模型输出结果进行反变换，得到最终结果
        y_test_unscaled = y_scaler.inverse_transform(y_test.reshape(1, -1)).squeeze()
        y_train_unscaled = y_scaler.inverse_transform(y_train.reshape(1, -1)).squeeze()
        y_pred_test_unscaled = y_scaler.inverse_transform(y_pred_test.reshape(1, -1)).squeeze()
        y_pred_train_unscaled = y_scaler.inverse_transform(y_pred_train.reshape(1, -1)).squeeze()

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
target = "temperature"  # Here output of the ML model in time

# 读取csv特征文件
df = pd.read_csv(
    "example.csv"
)  ## The csv file containing the input output of the ML models

# 模型训练和测试
train(df, target)  # call the function defined above to train the ML model
