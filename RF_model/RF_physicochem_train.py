import os

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import Sequence
import keras
from keras import regularizers

import numpy as np


def RF_six_training(train_desc_file, test_desc_file, save_path, project_name):
    """
    :param src_file: contain molecular name (ES_id or HS_id) and smile string
    :return:
    """
    train_desc_df = pd.read_csv(train_desc_file)
    train_desc = train_desc_df[['MolWt', 'TPSA', 'nRotB', 'HBD', 'HBA', 'LogP']]
    train_value = train_desc_df['targets']

    test_desc_df = pd.read_csv(test_desc_file)
    test_desc = test_desc_df[['MolWt', 'TPSA', 'nRotB', 'HBD', 'HBA', 'LogP']]
    test_value = test_desc_df['targets']

    clf = RandomForestClassifier(n_estimators=200, max_depth=None, oob_score=True, n_jobs=100)
    clf.fit(train_desc, train_desc_df['targets'])
    model_path = os.path.join(save_path, project_name+'_six_desciptor.m')
    joblib.dump(clf, model_path)

    y_prob = clf.predict_proba(test_desc)
    pred = clf.predict(test_desc)
    test_desc_df['class'] = pred
    test_desc_df['prob'] = y_prob[:, 1]
    test_desc_df.to_csv(os.path.join(save_path, 'test_predict.csv'))
    accuracy = clf.score(test_desc, test_value)
    matt = matthews_corrcoef(test_value, pred)
    fpr, tpr, threshold = roc_curve(test_value, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    # stats_list.append({'pred': pred,
    #                     'accuracy': accuracy,
    #                     'matthews_corrcoef': matt,
    #                     'roc_auc': roc_auc,
    #                     'n_estimator': n_estimator})
    print('Done')

if __name__ == '__main__':
    #### all data #####
    import argparse
    parser = argparse.ArgumentParser(description='Train Random Forest Model')
    parser.add_argument('--train_file', type=str, help='Specify the train file')
    # parser.add_argument('--val_file', type=str, help='Specify the val file')
    parser.add_argument('--test_file', type=str, help='Specify the test file')
    parser.add_argument('--save_path', type=str, help='Specify the result directory')
    parser.add_argument('--project_name', type=str, help='Specify the project name')

    args = parser.parse_args([
        '--train_file', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/60w_train_six_desc.csv',
        '--test_file', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/60w_test_six_desc.csv',
        '--save_path', '/data/baiqing/PycharmProjects/YaSAScore/data/RF_data/split_by_2_physicochem',
        '--project_name', 'split_2_physicochem',

    ])

    # args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    RF_six_training(args.train_file, args.test_file, args.save_path, args.project_name)

