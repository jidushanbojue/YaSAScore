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


def cal_descriptor(in_file, result_file):
    df = pd.read_csv(in_file)
    molwt_list = []
    tpsa_list = []
    nrotb_list = []
    hbd_list = []
    hba_list = []
    logp_list = []
    new_df = pd.DataFrame()
    for idx, line in df.iterrows():
        print(idx)
        mol = Chem.MolFromSmiles(line['smiles'])
        molwt_list.append(Descriptors.ExactMolWt(mol))
        tpsa_list.append(Chem.rdMolDescriptors.CalcTPSA(mol))
        nrotb_list.append(Descriptors.NumRotatableBonds(mol))
        hbd_list.append(Descriptors.NumHDonors(mol))
        hba_list.append(Descriptors.NumHAcceptors(mol))
        logp_list.append(Descriptors.MolLogP(mol))
    df['MolWt'] = molwt_list
    df['TPSA'] = tpsa_list
    df['nRotB'] = nrotb_list
    df['HBD'] = hbd_list
    df['HBA'] = hba_list
    df['LogP'] = logp_list
    df.to_csv(result_file)
    return np.array(new_df), np.array(df['targets'])



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate six descriptors')

    parser.add_argument('--in_file', type=str, help='Specify the absolute path to in file')
    parser.add_argument('--out', type=str, help='Specify the absolute property path to the descriptor files')

    # args = parser.parse_args([
    #     '--in_file', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/24w_train_df_seed0.csv',
    #     '--out', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/24w_train_six_desc.csv'
    # ])
    args = parser.parse_args()


    cal_descriptor(args.in_file, args.out)
