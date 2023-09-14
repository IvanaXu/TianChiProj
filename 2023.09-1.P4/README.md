## 文件结构与内容

Fingerprint_features和RAC_features分别代表两种不同的特征类型。

`./Fingerprint_features/`和`./RAC_features/`文件夹中的每个子文件夹分别是一个预测任务。
例如`./Fingerprint_features/temperature/`文件夹即表示模型预测内容为Fingerprint_features中的temperature特征。

需要注意的是Fingerprint_features特征中的linker1smi特征为字符串特征。可以考虑使用rdkit库chem模块中的`chem.RDKFingerprint`函数将其转换为数字向量特征。

`./Fingerprint_features/`中temperature和time均为单值预测回归任务。
`./RAC_features/`中temperature和time为单值预测回归任务，solvent为多值预测回归任务，additive为分类任务。

`test_RF.py`文件为使用随机森林模型。
`test_pytorch.py`文件为使用pytorch神经网络模型，其中给出了1D CNN、LSTM和RNN三种基本网络模型。

`read_and_calculate.py`文件可以合并保存下来的十折交叉验证的结果，计算总的评价指标结果。

**代码文件及详细注释内容请参考**：
`.\Fingerprint_features\temperature\`文件夹内代码
`.\RAC_features\time\`文件夹内代码

## 运行test代码

首先需要确保已安装所需的pytorch库:
`rdkit`
` scikit-learn`
`yaml`
`pandas`
`torch` (使用pytorch神经网络模型)

然后在对应文件夹路径下（`.\Fingerprint_features\temperature\`或`.\RAC_features\time\`），执行`python test_RF.py`或`python test_pytorch.py`即可。



------------------------------------------------------------

## File Structure and Contents

The folders "Fingerprint_features" and "RAC_features" represent two different types of feature sets.

Each subfolder in the `./Fingerprint_features/` and `./RAC_features/` directories corresponds to a specific prediction task. For instance, the `./Fingerprint_features/temperature/` folder indicates that the model's predictions are based on the "temperature" feature from the Fingerprint_features.

It is important to note that the "linker1smi" feature in the Fingerprint_features is a string-based feature. Consider using the `chem.RDKFingerprint` function from the rdkit library's chem module to convert it into a numerical vector feature.

In the `./Fingerprint_features/` directory, "temperature" and "time" are single-value prediction regression tasks.
In the `./RAC_features/` directory, "temperature" and "time" are single-value prediction regression tasks, "solvent" is a multi-value prediction regression task, and "additive" is a classification task.

The file `test_RF.py` uses the random forest model for prediction.
The file `test_pytorch.py` uses the PyTorch neural network model and provides three basic network models: 1D CNN, LSTM, and RNN.

The file `read_and_calculate.py` can merge and save the results from ten-fold cross-validation and calculate the overall evaluation metrics.

**For detailed code comments, please refer to the contents of**:
Code inside the `.\Fingerprint_features\temperature\` folder
Code inside the `.\RAC_features\time\` folder

## Running the Test Code

Ensure that the required PyTorch libraries are installed:
- `rdkit`
- `scikit-learn`
- `yaml`
- `pandas`
- `torch` (for using PyTorch neural network model)

Then, navigate to the corresponding folder path (`.\Fingerprint_features\temperature\` or `.\RAC_features\time\`) and execute `python test_RF.py` or `python test_pytorch.py` as needed.

