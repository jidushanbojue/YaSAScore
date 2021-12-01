import os
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
import numpy as np
from rdkit import Chem
from rdkit.DataStructs import cDataStructs
from rdkit.Chem import AllChem
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import Sequence
import keras
import tensorflow as tf
from keras import regularizers
import joblib
# from pathlib import Path

def predict(model, test_file, save_path, project_name):

    test_desc_df = pd.read_csv(test_file)


    policy = tf.keras.models.load_model(model)

    test_arr = np.array(test_desc_df[['MolWt', 'TPSA', 'nRotB', 'HBD', 'HBA', 'LogP']])
    test_value = keras.utils.to_categorical(np.array(test_desc_df['targets']).reshape(len(test_desc_df), 1), num_classes=2)


    pred_arr = policy.predict(x=test_arr, verbose=1)

    pred = np.argmax(pred_arr, axis=1)

    test_desc_df['class'] = pred
    test_desc_df.to_csv(os.path.join(save_path, project_name+'_predict.csv'))


    c = confusion_matrix(test_value[:, 1], pred)
    matt = matthews_corrcoef(test_value[:, 1], pred)
    acc = accuracy_score(test_value[:, 1], pred)

    fpr, tpr, threshold = roc_curve(test_value[:, 1], pred_arr[:, 1])
    roc_auc = auc(fpr, tpr)

    font1 = {'family': 'Nimbus Roman',
             'weight': 'normal',
             'size': 23}

    font2 = {'family': 'Nimbus Roman',
             'weight': 'normal',
             'size': 25}

    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='red', lw=5)
    plt.xlabel('False positive rate', fontdict=font1)
    plt.ylabel('True positive rate', fontdict=font1)
    plt.title('DNN classifier (AUC={})'.format(roc_auc), fontdict=font2)
    # plt.legend('R^2 of test: {}'.format(test_score))
    # plt.text(0.5, 0.6, 'Test score is {}'.format(accuracy))
    plt.text(0.5, 0.5, 'matthews corrcoef is {}'.format(matt))
    plt.text(0.5, 0.4, 'accuracy score is {}'.format(acc))
    plt.text(0.5, 0.3, 'confusion corrcoef is {}'.format(c))

    picture_path = os.path.join(save_path, project_name+'.png')

    plt.savefig(picture_path)
    plt.show()

    print('Done!!!')



if __name__ == '__main__':


    #### all data #####

    # base_dir = 'projects/data'
    # script_dir = os.path.dirname(os.path.abspath(__file__))

    import argparse
    parser = argparse.ArgumentParser(description='predict test file result from DNN ECFP model')
    parser.add_argument('--model_path', type=str, help='Specify the absolute path of model path')
    parser.add_argument('--test_file', type=str, help='Specify the test file')
    parser.add_argument('--save_path', type=str, help='Specify the result directory')
    parser.add_argument('--project_name', type=str, help='Specify the project name')
    parser.add_argument('--gpu_index', type=str, help='Specify the GPU index to use')

    # args = parser.parse_args([
    #     '--model_path', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/split_by_4_physicochem/split_4_physicochem.hdf5',
    #     '--test_file', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/8w_test_six_desc.csv',
    #     '--save_path', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/split_by_4_physicochem',
    #     '--project_name', 'split_4_physicochem_test',
    #       '--gpu_index', '0'
    # ])

    args = parser.parse_args()

    ##### ECFP  ####

    test_file = args.test_file
    model_path = args.model_path
    save_path = args.save_path
    project_name = args.project_name

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    predict(model=model_path,
            test_file=test_file,
            save_path=save_path,
            project_name=project_name)