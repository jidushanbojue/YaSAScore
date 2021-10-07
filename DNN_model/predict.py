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


def smiles_to_ecfp(product, size=2048):
    """Converts a single SMILES into an ECFP4
    :parameter:
        product (str): The SMILES string corresponding to the product/molecule of choice.
        size (int): Size (dimension) of the ECFP4 vector to be calculated.
    Return:
        ecfp4 (arr): An n dimensional ECFP4 vector of the molecule
    """
    try:
        mol = Chem.MolFromSmiles(product)
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=size)
        arr = np.zeros((0,), dtype=np.int8)
        cDataStructs.ConvertToNumpyArray(ecfp, arr)
    except:
        arr = ''
    return arr


class FPSequnce(Sequence):
    def __init__(self, input_matrix, label_matrix, batch_size):
        self.input_matrix = input_matrix
        self.label_matrix = label_matrix
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.label_matrix.shape[0] / float(self.batch_size)))

    def __getitem__(self, item):
        X_input = self.input_matrix[item*self.batch_size: (item+1)*self.batch_size]
        Y_input = self.label_matrix[item*self.batch_size: (item+1)*self.batch_size]

        return (X_input.todense(), Y_input)
        # return (X_input, Y_input)

def predict(model, test_file, save_path, project_name):
    policy = tf.keras.models.load_model(model)
    test_inputs, test_labels_one_hot = generate_dataset(test_file)


    predict_arr = policy.predict(x=FPSequnce(test_inputs, test_labels_one_hot, batch_size=1), verbose=1)

    test_df = pd.read_csv(test_file)
    test_df['HS_rate'] = predict_arr[:, 0]
    test_df['ES_rate'] = predict_arr[:, 1]

    pred = np.argmax(predict_arr, axis=1)
    test_df['class'] = pred
    test_df.to_csv(os.path.join(save_path, project_name+'_predict.csv'))

    c = confusion_matrix(test_labels_one_hot[:, 1], pred)
    matt = matthews_corrcoef(test_labels_one_hot[:, 1], pred)

    fpr, tpr, threshold = roc_curve(test_labels_one_hot[:, 1], predict_arr[:, 1])
    roc_auc = auc(fpr, tpr)

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23}

    font2 = {'family': 'Times New Roman',
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
    # plt.text(0.5, 0.4, 'accuracy score is {}'.format(accuracy))
    plt.text(0.5, 0.3, 'confusion corrcoef is {}'.format(c))

    # plt.savefig('ECFP_from_all_data_RF_roc_AUC1_n_estimator1000_oobTrue_mean.png')
    picture_path = os.path.join(save_path, project_name+'.png')

    plt.savefig(picture_path)
    plt.show()

    print('Done!!!')


def generate_dataset(src_file):
    """
    :param src_file: containing the smiles and targets
    :return: data_input, data_label
    """
    df = pd.read_csv(src_file)
    # df.columns = ['id', 'smiles', 'targets']
    input_list = [x for x in df['smiles'].apply(smiles_to_ecfp, size=2048).values]
    input_csr = sparse.csr_matrix(input_list)
    input_label = keras.utils.to_categorical(np.array(df['targets']).reshape(len(df), 1), num_classes=2)
    print('Done')
    return input_csr, input_label


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
    #     '--model_path', '/data/baiqing/PycharmProjects/dnn_model/split_by_4/split_4.hdf5',
    #     '--test_file', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/8w_cmpnn_remain_all_test_4_split.csv',
    #     '--save_path', '/data/baiqing/PycharmProjects/dnn_model/split_by_4',
    #     '--project_name', 'split_4_all_test',
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

    # train_file = os.path.join(base_dir, '24w_train_df_seed0.csv')
    # val_file = os.path.join(base_dir, '24w_val_df_seed0.csv')

    ### test set from balanced set
    # test_file = os.path.join(base_dir, '24w_test_df_seed0.csv')

    ### test set from imbalanced set
    # test_file = os.path.join(base_dir, '24w_cmpnn_remain_all_test.csv')

    # result_picture = os.path.join(script_dir, '19w_min_ecfp_3_3_0.0001_3layers_all_test.png')


    # DNN_ECFP_training(train_file, val_file, save_path, project_name)
    ######################

    predict(model=model_path,
            test_file=test_file,
            save_path=save_path,
            project_name=project_name)
