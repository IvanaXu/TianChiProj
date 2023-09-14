import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import time

disease_mapping = {
    'control': 0,
    "Alzheimer's disease": 1,
    "Graves' disease": 2,
    "Huntington's disease": 3,
    "Parkinson's disease": 4,
    'rheumatoid arthritis': 5,
    'schizophrenia': 6,
    "Sjogren's syndrome": 7,
    'stroke': 8,
    'type 2 diabetes': 9
}
sample_type_mapping = {'control': 0, 'disease tissue': 1}


def load_idmap(idmap_dir):
    idmap = pd.read_csv(idmap_dir, sep=',')
    age = idmap.age.to_numpy()
    age = age.astype(np.float32)
    sample_type = idmap.sample_type.replace(sample_type_mapping)
    return age, sample_type


def load_methylation(methy_dir):
    '''
    Load methylation data from csv file.

    Note: We set nrows=5000 for test.
    If you want to use full data, it is recommended to read csv file by chunks 
      or other methods since the csv file is very large.
    Note the memory usage when you read csv file.

    We fill nan with 0, you can try other methods.
    '''

    methylation = pd.read_csv(methy_dir, sep=',', index_col=0, nrows=5000)
    methylation.fillna(0, inplace=True)
    methylation = methylation.values.T.astype(np.float32)
    return methylation


def load_methylation_h5(prefix):
    '''
    Load methylation data from .h5 file. 

    Parameters:
    ------------
    prefix: 'train' or 'test'
    '''
    methylation = h5py.File(prefix + '.h5', 'r')['data']
    h5py.File(prefix + '.h5', 'r').close()
    return methylation[:, :5000]  # 5000 just for test
    return methylation[:, :]  # If you want to use full data, you can use this line.


def train_ml(X_train, y_train):
    model = ElasticNet()
    model.fit(X_train, y_train)
    return model


def evaluate_ml(y_true, y_pred, sample_type):
    '''
    This function is used to evaluate the performance of the model. 

    Parameters:
    ------------
    y_true: true age
    y_pred: predicted age
    sample_type: sample type, 0 for control, 1 for case
    
    Return:
    ------------
    mae: mean absolute error.
    mae_control: mean absolute error of control samples.
    mae_case: mean absolute error of case samples.

    We use MAE to evaluate the performance.
    Please refer to evaluation section in the the official website for more details.
    '''
    mae_control = np.mean(
        np.abs(y_true[sample_type == 0] - y_pred[sample_type == 0]))

    case_true = y_true[sample_type == 1]
    case_pred = y_pred[sample_type == 1]
    above = np.where(case_pred >= case_true)
    below = np.where(case_pred < case_true)

    ae_above = np.sum(np.abs(case_true[above] - case_pred[above])) / 2
    ae_below = np.sum(np.abs(case_true[below] - case_pred[below]))
    mae_case = (ae_above + ae_below) / len(case_true)

    mae = np.mean([mae_control, mae_case])
    return mae, mae_control, mae_case


if __name__ == "__main__":

    idmap_train_dir = 'trainmap.csv'
    methy_train_dir = 'traindata.csv'
    idmap_test_dir = 'testmap.csv'
    methy_test_dir = 'testdata.csv'

    age, sample_type = load_idmap(idmap_train_dir)

    # Note: 'traindata.csv' is about 57GB, 'testdata.csv' is about 15GB.
    # If you want to use h5 file,  you must run data_h5.py first to generate .h5 file.
    # 'train.h5' is about 15GB, 'test.h5' is about 3.8GB.
    # However, the memory usage is still large when you load .h5 file.
    # Using this code directly on the free server provided by Tianchi will still
    # result in insufficient memory when training ElasticNet on the full dataset.

    use_h5 = True
    if use_h5:
        methylation = load_methylation_h5('train')
        methylation_test = load_methylation_h5('test')
    else:
        methylation = load_methylation(methy_train_dir)
        methylation_test = load_methylation(methy_test_dir)
    print('Load data done')

    indices = np.arange(len(age))
    [indices_train, indices_valid, age_train,
     age_valid] = train_test_split(indices, age, test_size=0.3, shuffle=True)
    methylation_train, methylation_valid = methylation[
        indices_train], methylation[indices_valid]
    sample_type_train, sample_type_valid = sample_type[
        indices_train], sample_type[indices_valid]
    feature_size = methylation_train.shape[1]
    del methylation

    print('Start training...')
    start = time.time()
    pred_model = train_ml(methylation_train, age_train)
    print(f'Training time: {time.time() - start}s')

    age_valid_pred = pred_model.predict(methylation_valid)
    mae = evaluate_ml(age_valid, age_valid_pred, sample_type_valid)
    print(f'Validation MAE: {mae}')

    age_pred = pred_model.predict(methylation_test)

    age_pred[age_pred < 0] = 0  
    # naive post-processing to ensure age >= 0

    age_pred = np.around(age_pred, decimals=2)
    age_pred = ['%.2f' % i for i in age_pred]
    sample_id = pd.read_csv(idmap_test_dir, sep=',').sample_id
    # Note: sample_id in submission should be the same as the order in testmap.csv.
    # We do not provide the matching producdure for disordered sample_id in evaluation.

    submission = pd.DataFrame({'sample_id': sample_id, 'age': age_pred})
    submission_file = 'submit.txt'
    submission.to_csv(submission_file, index=False)
    