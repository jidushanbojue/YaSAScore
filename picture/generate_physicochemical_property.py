import os
import seaborn as sns
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen


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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot distribution picture of different data set')
    # base_dir = 'projects/data'

    parser.add_argument('--ES_file', type=str, help='Specify the absolute path to the ES-file')
    parser.add_argument('--HS_file', type=str, help='Specify the absolute path to the HS-file')
    parser.add_argument('--out', type=str, help='Specify the absolute property path to the property')
    # parser.add_argument('--picture_out', type=str, help='Specify the absolute picture path')


    # args = parser.parse_args([
    #     '--ES_file', '/data/baiqing/PycharmProjects/YaSAScore/data/cmpnn_data/24w_ES.csv',
    #     '--HS_file', '/data/baiqing/PycharmProjects/YaSAScore/data/cmpnn_data/24w_HS.csv',
    #     '--out', '/data/baiqing/PycharmProjects/YaSAScore/data/cmpnn_data/24w_ES_HS_property.csv'
    # ])

    args = parser.parse_args()

    property_generate(args.ES_file, args.HS_file, args.out)


