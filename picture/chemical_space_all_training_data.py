import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns


def smiles_to_ecfp(product, size=2048):
    """Converts a single SMILES into an ECFP4

    Parameters:
        product (str): The SMILES string corresponing to the product/molecule of choice.
        size (int): Size (dimensions) of the ECFP4 vector to be calculated.

    Returns:
        ecfp4 (arr): An n dimensional ECFP4 vector of the molecule.
    """
    mol = Chem.MolFromSmiles(product)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=size)
    arr = np.zeros((0,), dtype=np.int8)
    cDataStructs.ConvertToNumpyArray(ecfp, arr)
    return arr


def cal_descriptor(train_file, val_file, test_file, result_file):
    # df = pd.read_csv(src_file)
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)

    df = pd.concat([df_train, df_val, df_test])
    print('whole molecules: ', len(df))
    molwt_list = []
    tpsa_list = []
    nrotb_list = []
    hbd_list = []
    hba_list = []
    logp_list = []
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

    descriptors = df.loc[:, ['MolWt', 'TPSA', 'nRotB', 'HBD', 'HBA', 'LogP']].values
    descriptors_std = StandardScaler().fit_transform(descriptors)


    tsne = manifold.TSNE(n_components=2, init='pca', random_state=42, n_jobs=160)
    tsne_arr = tsne.fit_transform(descriptors_std)
    tsne_df = pd.DataFrame(tsne_arr, columns=['t-SNE1', 't-SNE2'])
    # result = pd.concat([merged_df, tsne_df], axis=1)
    df['t-SNE1'] = tsne_df['t-SNE1']
    df['t-SNE2'] = tsne_df['t-SNE2']

    pca = PCA()
    descriptors_2d = pca.fit_transform(descriptors_std)
    descriptors_pca = pd.DataFrame(descriptors_2d)
    descriptors_pca.index = df.index
    descriptors_pca.columns = ['PC{}'.format(i+1) for i in descriptors_pca.columns]
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))

    result_df = pd.concat([df, descriptors_pca], axis=1)
    result_df.to_csv(result_file)
    print('Done')


def plot_pca_tsne(src_file, threshold, result_pca):
    df = pd.read_csv(src_file)
    # df_HS = df[df['label'] == 0].sample(n=120000, random_state=42)
    # df_ES = df[df['label'] == 1].sample(n=120000, random_state=42)

    df_HS = df[df['targets'] == 0].sample(n=5000, random_state=42)
    df_ES = df[df['targets'] == 1].sample(n=5000, random_state=42)

    plt.figure(figsize=(8, 8), dpi=600)
    plt.scatter(df_HS['PC1'], df_HS['PC2'], c='g', marker='^', s=20, label='HS')
    plt.scatter(df_ES['PC1'], df_ES['PC2'], c='r', marker='*', s=20, label='ES')
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.legend(fontsize=15)

    font_title = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 15}

    font_axis = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 12}


    plt.xlabel('PC1', font_axis)
    plt.ylabel('PC2', font_axis)
    # plt.savefig('24w_last_PCA.png')
    plt.title('PCA analysis of ES:HS dataset (split by {} Reaction Steps)'.format(threshold), font_title)

    plt.savefig(result_pca)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot distribution picture of different data set')
    # base_dir = 'projects/data'

    # parser.add_argument('--train_file', type=str, help='Specify the absolute path to the train-file')
    # parser.add_argument('--val_file', type=str, help='Specify the absolute path to the val-file')
    # parser.add_argument('--test_file', type=str, help='Specify the absolute path to the test-file')
    parser.add_argument('--pca_result', type=str, help='Specify the absolute path to the pca result file')
    parser.add_argument('--threshold', type=str, help='can be 2, 3, 4')
    parser.add_argument('--out', type=str, help='Specify the picture path')

    # args = parser.parse_args([
    #     '--train_file', '../data/cmpnn_data/8w_train_df_seed0.csv',
    #     '--val_file', '../data/cmpnn_data/8w_val_df_seed0.csv',
    #     '--test_file', '../data/cmpnn_data/8w_test_df_seed0.csv',
    #     '--pca_result', '../data/cmpnn_data/8w_pca_result.csv',
    #     '--threshold', 4,
    #     '--out', '8w_pca_picture.png'
    # ])

    args = parser.parse_args()

    # cal_descriptor(args.train_file, args.val_file, args.test_file, args.pca_result)
    plot_pca_tsne(args.pca_result, args.threshold, args.out)

