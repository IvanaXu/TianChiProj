# https://tianchi.aliyun.com/forum/postDetail?spm=a2c22.12586969.0.0.549a5bf6c9EBee&postId=588511

import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import pandas as pd  # 导入Pandas库，用于数据处理和分析
import xarray as xr  # 导入Xarray库，用于处理标签化的多维数据
from collections import defaultdict, Counter  # 导入defaultdict和Counter类，用于创建默认字典和计数器
import xgboost as xgb  # 导入XGBoost库，用于梯度提升树模型
import lightgbm as lgb  # 导入LightGBM库，也用于梯度提升树模型
from catboost import CatBoostRegressor  # 导入CatBoostRegressor类，用于CatBoost回归模型
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold  # 导入三种交叉验证类，分别是分层K折交叉验证、K折交叉验证和分组K折交叉验证
from sklearn.metrics import mean_squared_error  # 导入均方误差函数
from joblib import Parallel, delayed  # 导入Parallel和delayed类，用于并行化计算任务
from tqdm.auto import tqdm  # 导入tqdm类，用于在循环中显示进度条
import sys, os, gc, argparse, warnings, torch, glob  # 导入其他常用模块，如sys、os、gc、argparse、warnings、torch、glob

warnings.filterwarnings('ignore')  # 设置警告过滤器，将忽略警告信息

# pip install importlib-metadata==4.13.0
# pip install zarr lightgbm catboost
# pip install xarray[complete]

path = './weather'  # 数据集文件路径

def chunk_time(ds):
    # 获取数据集维度的字典并赋值给dims变量
    dims = {k: v for k, v in ds.dims.items()}
    # 在'time'维度上将数据集进行分块处理，每块包含一个时间步
    dims['time'] = 1
    ds = ds.chunk(dims)
    return ds

def load_dataset():
    ds = []  
    for y in range(2007, 2012):  # 遍历年份2007到2011
        data_name = os.path.join(path, f'weather_round1_train_{y}')  # 构建数据文件名
        x = xr.open_zarr(data_name, consolidated=True)  # 打开Zarr格式的数据集文件
        print(f'{data_name}, {x.time.values[0]} ~ {x.time.values[-1]}')  # 输出数据文件名和时间范围
        ds.append(x)  # 将打开的数据集添加到ds列表中
        
    ds = xr.concat(ds, 'time')  # 在'time'维度上连接所有数据集
    ds = chunk_time(ds)  # 对数据集进行时间分块处理
    return ds

ds = load_dataset().x  # 调用load_dataset函数加载数据集，并获取其中的'x'变量

lats = ds.lat.values.tolist()  # 将数据集中的纬度值转换为Python列表，并赋值给lats变量
lons = ds.lon.values.tolist()  # 将数据集中的经度值转换为Python列表，并赋值给lons变量

# 对齐测试数据，每份数据应该是22个时刻
# 获取多份训练数据，滑动5次，每次24个时刻（6天），每次滑动提取最后22个时刻作为一份训练数据
train_data = []
for i in range(5):
    if i == 0:  # 如果是第一份训练数据
        print(-22, 0)  # 输出起始和结束索引
        train_data.append(ds[-22:, :, :, :].values)  # 提取最后22个时刻的数据，添加到训练数据列表中
    else:
        idx = i * 24  # 计算索引偏移量
        train_data.append(ds[-22-idx:-idx, :, :, :].values)  # 提取最后22个时刻的数据，添加到训练数据列表中
        print(-22-idx, -idx)  # 输出起始和结束索引

# 经纬度标识字段
latlon_vals = []
for i in range(161):  # 遍历纬度索引
    for j in range(161):  # 遍历经度索引
        latlon_vals.append([lats[i], lons[j]])  # 将每个纬度和经度对应的值添加到列表中
latlon_vals = np.array(latlon_vals)  # 将列表转换为NumPy数组，并赋值给latlon_vals变量

# 确定好20份训练数据后，接下来需要提取特征
for flag, dat in tqdm(enumerate(train_data)):  # 遍历每份训练数据，并显示进度条
    # 第一时刻特征
    first_feat = dat[0,:,:,:].reshape(70,161*161).transpose()  # 将第一时刻的数据转换为特征向量，reshape后的形状为(70, 161*161)，然后进行转置得到(161*161, 70)
    # 第二时刻特征
    second_feat = dat[1,:,:,:].reshape(70,161*161).transpose()  # 将第二时刻的数据转换为特征向量，reshape后的形状为(70, 161*161)，然后进行转置得到(161*161, 70)
    # 两个时刻差分特征
    diff_feat = (dat[1,:,:,:] - dat[0,:,:,:]).reshape(70,161*161).transpose()  # 将两个时刻的数据差分并转换为特征向量，reshape后的形状为(70, 161*161)，然后进行转置得到(161*161, 70)
    
    # 构建训练样本
    tmp_dat = dat[2:,-5:,:,:]  # 从第3个时刻开始获取最后5个channels的数据，形状为(20, 5, 161, 161)
    for i in range(20):  # 遍历20个时刻
        # 时间特征、经纬特征
        time_vals = np.array([i]*161*161).reshape(161*161,1)  # 创建一个维度为(161*161,1)的数组，每个元素为当前时刻的时间特征i
        sub_vals = np.concatenate((time_vals, latlon_vals), axis=1)  # 将时间特征和经纬度特征拼接在一起，形状为(161*161,3)
        
        # 添加历史特征，第一时刻特征、第二时刻特征、两个时刻差分特征
        sub_vals = np.concatenate((sub_vals, first_feat), axis=1)  # 将第一时刻特征拼接到子特征向量中，形状变为(161*161,73)
        sub_vals = np.concatenate((sub_vals, second_feat), axis=1)  # 将第二时刻特征拼接到子特征向量中，形状变为(161*161,143)
        sub_vals = np.concatenate((sub_vals, diff_feat), axis=1)  # 将两个时刻差分特征拼接到子特征向量中，形状变为(161*161,213)
        
        # 添加5个目标变量，形状变为(161*161,218)
        for j in range(5):
            var_vals = tmp_dat[i,j,:,:].reshape(161*161, 1)  # 将目标变量的数据转换为特征向量，形状为(161*161,1)
            sub_vals = np.concatenate((sub_vals, var_vals), axis=1)  # 将目标变量拼接到子特征向量中
        
        if i == 0 :
            all_vals = sub_vals  # 如果是第一个时刻，则将子特征向量赋值给all_vals
        else:
            all_vals = np.concatenate((all_vals, sub_vals), axis=0)  # 如果不是第一个时刻，则将子特征向量与all_vals进行垂直拼接
        
    if flag == 0:
        final_vals = all_vals  # 如果是第一份训练数据，则将all_vals赋值给final_vals
    else:
        final_vals = np.concatenate((final_vals, all_vals), axis=0)  # 如果不是第一份训练数据，则将all_vals与final_vals进行垂直拼接

train_df = pd.DataFrame(final_vals)  # 将提取的特征final_vals转换为DataFrame格式，并赋值给train_df
train_df.columns = ['time','lat','lon'] + [f'feat_{i}' for i in range(210)] + ['t2m','u10','v10','msl','tp']
# 添加列名，分别为时间特征、经度特征、纬度特征、210个特征向量、目标变量t2m、u10、v10、msl和tp

cols = ['time','lat','lon'] + [f'feat_{i}' for i in range(210)]
# 创建列名列表，包括时间特征、经度特征、纬度特征和210个特征向量列名


def train_model(train_x, train_y, label, seed=2023):
    # 初始化交叉验证的变量和参数
    oof = np.zeros(train_x.shape[0])  # 存储交叉验证的预测结果
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)  # 5折交叉验证
    cv_scores = []  # 存储每一折的验证集分数

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} {}************************************'.format(str(i+1), str(seed)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        #trn_x, trn_y, val_x, val_y：根据索引划分出当前折的训练集和验证集。

        # 设置CatBoost模型的参数
        params = {'learning_rate': 0.5,
                  'depth': 5,
                  'bootstrap_type':'Bernoulli',
                  'random_seed':2023,
                  'od_type': 'Iter',
                  'od_wait': 100,
                  'random_seed': 11,
                  'allow_writing_files': False,
                  'task_type':"GPU",
                  'devices':'0:1'}

        model = CatBoostRegressor(iterations=100, **params)  # 创建CatBoost回归模型
        model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                  cat_features=[],
                  use_best_model=True,
                  verbose=1)  # 在训练集上拟合模型并在验证集上进行评估
        # model.save_model(f'catboost_{label}_fold{i}.model')  # 保存模型
        model.save_model(f'model/catboost_{label}_fold{i}.model')  # 保存模型到指定路径
        val_pred  = model.predict(val_x)  # 在验证集上进行预测
        oof[valid_index] = val_pred  # 将验证集的预测结果填充到oof数组中

        cv_scores.append(np.sqrt(mean_squared_error(val_y, val_pred)))  # 计算并存储验证集的均方根误差
        
        if i == 0: #如果是第一折（i==0），计算特征重要性并保存到CSV文件。
            fea_ = model.feature_importances_
            fea_name = model.feature_names_
            fea_score = pd.DataFrame({'fea_name':fea_name, 'score':fea_})
            fea_score = fea_score.sort_values('score', ascending=False)
            fea_score.to_csv('feature_importances.csv', index=False)

        print(cv_scores) #打印当前折的验证集分数。
    return oof
    
oofs = []  # 存储所有特征的交叉验证预测结果

# 对每个特征进行训练和预测
for label in ['t2m', 'u10', 'v10', 'msl', 'tp']:
    print(f'==================== {label} ====================')
    
    # 调用train_model函数进行训练，并将返回的交叉验证预测结果存储到cat_oof变量中
    cat_oof = train_model(train_df[cols], train_df[label], label)
    
    # 将交叉验证预测结果添加到oofs列表中
    oofs.append(cat_oof)

def inter_model(test_x, label, seed=2023):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)  # 5折交叉验证
    test = np.zeros(test_x.shape[0])  # 存储测试集的预测结果

    params = {'learning_rate': 0.5, 'depth': 6, 'bootstrap_type':'Bernoulli','random_seed':seed, 'metric_period': 300,
              'od_type': 'Iter', 'od_wait': 100, 'random_seed': 11, 'allow_writing_files': False, 'task_type': 'CPU',
              'task_type':"GPU", 'devices':'0:1'}

    for i in range(0, 5):
        # print('************************************ {} {}************************************'.format(str(i+1), str(seed)))
        model = CatBoostRegressor(**params)
        # model.load_model(f'catboost_{label}_fold{i}.model')
        model.load_model(f'model/catboost_{label}_fold{i}.model')  # 加载训练好的模型
        test_pred = model.predict(test_x)  # 在测试集上进行预测
        test += test_pred / kf.n_splits  # 将每一折的预测结果加权平均得到最终的测试集预测结果

    return test

def get_test_data(dat):
    # 第一时刻特征
    first_feat = dat[0,:,:,:].reshape(70,161*161).transpose() # 将第一时刻的特征数据从(70, 161, 161)转换为(161*161, 70)

    # 第二时刻特征
    second_feat = dat[1,:,:,:].reshape(70,161*161).transpose() # 将第二时刻的特征数据从(70, 161, 161)转换为(161*161, 70)

    # 两个时刻差分特征
    diff_feat = (dat[1,:,:,:] - dat[0,:,:,:]).reshape(70,161*161).transpose() # 计算两个时刻特征之间的差分，并将结果从(70, 161, 161)转换为(161*161, 70)
    
    # 构建测试样本
    for i in range(20): # 20个时刻
        # 时间特征、经纬特征
        time_vals = np.array([i]*161*161).reshape(161*161,1) # 创建一个形状为(161*161, 1)的数组，每个元素都是i
        sub_vals = np.concatenate((time_vals, latlon_vals), axis=1) # 将时间特征和经纬度特征按列进行拼接，形状为(161*161, 3)
        
        # 添加历史特征，第一时刻特征、第二时刻特征、两个时刻差分特征
        sub_vals = np.concatenate((sub_vals, first_feat), axis=1) # 将第一时刻特征拼接到sub_vals中，形状为(161*161, 73)
        sub_vals = np.concatenate((sub_vals, second_feat), axis=1) # 将第二时刻特征拼接到sub_vals中，形状为(161*161, 143)
        sub_vals = np.concatenate((sub_vals, diff_feat), axis=1) # 将两个时刻差分特征拼接到sub_vals中，形状为(161*161, 213)
        
        if i == 0 :
            all_vals = sub_vals
        else:
            all_vals = np.concatenate((all_vals, sub_vals), axis=0) # 将sub_vals沿着行方向进行拼接
            
    df = pd.DataFrame(all_vals) # 将结果转换为DataFrame格式
    df.columns = ['time','lat','lon'] + [f'feat_{i}' for i in range(210)] # 给DataFrame的列命名
    return df

def get_parallel_result(file):
    # 加载输入数据
    input_data = torch.load(file)  # 加载保存在文件中的输入数据

    # 生成测试样本，并构建特征
    test_df = get_test_data(np.array(input_data))  # 根据输入数据生成测试样本并构建特征

    # 保存结果
    res = test_df[['time']]  # 创建一个只包含时间列的DataFrame用于保存结果

    # 模型推理，预测测试集结果
    for label in ['t2m', 'u10', 'v10', 'msl', 'tp']:
        # 通过模型推理预测测试集结果，inter_model是一个未给出的模型推理函数
        cat_pred = inter_model(test_df[cols], label)
        res[label] = cat_pred  # 将预测结果保存到res中

    # 转为提交格式
    for label in ['t2m', 'u10', 'v10', 'msl', 'tp']:
        for i in range(20):
            sub_vals = res[res['time'] == i][label].values.reshape(1, 161, 161)
            # 将时间为i的某个标签的预测结果转换为形状为(1, 161, 161)的数组

            if i == 0:
                all_vals = sub_vals
            else:
                all_vals = np.concatenate((all_vals, sub_vals), axis=0)
                # 将所有时间步的预测结果按时间序列连接起来，形状为(20, 161, 161)

        all_vals = all_vals.reshape(20, 1, 161, 161)

        if label == 't2m':
            final_vals = all_vals
        else:
            final_vals = np.concatenate((final_vals, all_vals), axis=1)
            # 将不同标签的预测结果按照通道连接起来，形状为(20, 5, 161, 161)

    final_vals = torch.tensor(final_vals)  # 将最终的结果转换为PyTorch张量
    save_file = file.split('/')[-1]  # 获取保存结果的文件名

    # torch.save(final_vals.half(), f'./{save_file}')  # 将结果以半精度浮点类型保存到指定路径中
    torch.save(final_vals.half(), f'output/{save_file}')  # 将结果以半精度浮点类型保存到指定路径中

# 并行处理指定目录下的文件
Parallel(n_jobs=8)( #创建一个并行处理的上下文，指定使用 8 个工作线程。

    # 对每个文件路径调用 get_parallel_result 函数进行处理
    delayed(get_parallel_result)(file_path)
    #使用 delayed 函数将 get_parallel_result 函数及其参数 file_path 封装起来，表示要对每个文件路径调用 get_parallel_result(file_path) 进行处理
    
    # 遍历指定目录下以 .pt 为后缀的文件路径
    for file_path in tqdm(glob.glob(f'{path}/weather_round1_test/input/*.pt'))
    #使用glob.glob函数获取指定目录下所有以 .pt 结尾的文件路径，并使用 tqdm 函数在进度条中显示遍历的进度。
)

#将output文件夹压缩，如果这段代码无法生成output.zip，运行下面的代码
_ = !zip -r output.zip output/


#如果上面代码无响应，运行下面这段代码
import zipfile, os
path = './output/'  # 要压缩的文件路径
zipName = 'output.zip'  # 压缩后的zip文件路径及名称

# 创建一个新的zip文件
f = zipfile.ZipFile(zipName, 'w', zipfile.ZIP_DEFLATED)
#使用zipfile模块创建一个新的zip文件对象，指定为写模式('w')并采用ZIP_DEFLATED压缩算法。

# 遍历指定路径下的所有文件和文件夹
for dirpath, dirnames, filenames in os.walk(path): #使用os.walk函数遍历指定路径下的所有文件和文件夹，包括子文件夹
    for filename in filenames: #遍历每个文件夹中的文件名。
        print(filename)
        # 将文件添加到zip文件中
        f.write(os.path.join(dirpath, filename))

# 关闭zip文件
f.close()