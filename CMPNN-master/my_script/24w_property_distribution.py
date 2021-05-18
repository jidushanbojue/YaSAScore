import os
import seaborn as sns
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
import matplotlib.pyplot as plt

def cal_descriptor(df):
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
    return df

def property_generate(ES_file, HS_file, result_file):
    df_ES = pd.read_csv(ES_file)
    df_HS = pd.read_csv(HS_file)
    ES_labels = ['ES'] * len(df_ES)
    HS_labels = ['HS'] * len(df_HS)
    df_ES['labels'] = ES_labels
    df_HS['labels'] = HS_labels
    df = pd.concat([df_ES, df_HS], axis=0)
    df = cal_descriptor(df)
    df.to_csv(result_file)

    # sns.boxplot(df, x='MolWt', hue='labels')
    # plt.show()

    print('Done')

def sns_boxplot(file):
    df = pd.read_csv(file)

    fig, axes = plt.subplots(2, 3, figsize=(20, 15))
    # print(df.)
    sns.boxplot(y='MolWt', x='labels', data=df, ax=axes[0, 0])
    sns.boxplot(y='HBD', x='labels', data=df, ax=axes[0, 1])
    sns.boxplot(y='HBA', x='labels', data=df, ax=axes[0, 2])
    sns.boxplot(y='nRotB', x='labels', data=df, ax=axes[1, 0])
    sns.boxplot(y='LogP', x='labels', data=df, ax=axes[1, 1])
    sns.boxplot(y='TPSA', x='labels', data=df, ax=axes[1, 2])
    plt.savefig('property_boxplot_24w.png')

    plt.show()

def sns_kde(file):
    df = pd.read_csv(file)
    fig, axes = plt.subplots(2, 3, figsize=(20, 15))
    # sns.histplot(x='MolWt', data=df, hue='labels', ax=axes[0, 0], kde='True')
    sns.kdeplot(x='MolWt', data=df, hue='labels', ax=axes[0, 0], linewidth=4, shade=True, bw=0.3)
    # h, l = ax0.get_legend_handles_labels()
    axes[0, 0].legend(labels=['HS', 'ES'])
    # h, l = axes[0, 0].get_legend_handles_labels()
    # axes[0, 0].legend(handles=h, labels=['ES', 'HS'])
    # axes[0, 0].legend_.remove()
    # sns.histplot(x='MolWt', data=df, ax=axes[0, 0], hue='labels', kde=True, bins=20)
    sns.kdeplot(x='HBD', data=df, hue='labels', ax=axes[0, 1], linewidth=4, shade=True)
    axes[0, 1].legend(labels=['ES', 'HS'])
    # sns.histplot(x='HBD', data=df, ax=axes[0, 1], hue='labels', kde=False, bins=20)
    sns.kdeplot(x='HBA', data=df, hue='labels', ax=axes[0, 2], linewidth=4, bw=0.3, shade=True)
    axes[0, 2].legend(labels=['ES', 'HS'])
    sns.kdeplot(x='nRotB', data=df, hue='labels', ax=axes[1, 0], linewidth=4, bw=0.3, shade=True)
    axes[1, 0].legend(labels=['ES', 'HS'])
    sns.kdeplot(x='LogP', data=df, hue='labels', ax=axes[1, 1], linewidth=4, shade=True, bw=0.3)
    axes[1, 1].legend(labels=['ES', 'HS'])
    sns.kdeplot(x='TPSA', data=df, hue='labels', ax=axes[1, 2], linewidth=4, shade=True, bw=0.3)
    axes[1, 2].legend(labels=['ES', 'HS'])
    # fig.legend(labels=['ES', 'HS'], loc='upper center')


    # plt.legend()
    plt.savefig('property_kdeplot_24w.png')
    plt.show()

def sns_kde_plot(file):
    df = pd.read_csv(file)
    fig, axes = plt.subplots(2, 3, figsize=(20, 15))





if __name__ == '__main__':
    base_dir = '/CMPNN-master/data'
    ES_file = os.path.join(base_dir, '24w_train_ES.csv')
    HS_file = os.path.join(base_dir, '24w_train_HS.csv')
    print('Done')

    property_file = os.path.join(base_dir, '24w_property_ES_HS.csv')
    # boxplot_fig_file = os.path.join(base_dir, 'property_boxplot_24w.png')
    # kdeplot_fig_file = os.path.join(base_dir, 'property_kdeplot_24w.png')
    # property_generate(ES_file, HS_file, property_file)

    # sns_boxplot(file=property_file, fig_file=boxplot_fig_file)
    sns_kde(property_file)
    # sns_boxplot(property_file)