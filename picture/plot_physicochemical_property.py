import os
import seaborn as sns
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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

def sns_kde(file, threshold, result):
    df = pd.read_csv(file)
    fig, axes = plt.subplots(2, 3, figsize=(20, 15))

    # sns.set(font_scale=5)

    # sns.histplot(x='MolWt', data=df, hue='labels', ax=axes[0, 0], kde='True')
    sns.kdeplot(x='MolWt', data=df, hue='labels', ax=axes[0, 0], linewidth=4, shade=True, bw=0.3)
    # h, l = ax0.get_legend_handles_labels()
    axes[0, 0].legend(labels=['HS', 'ES'], fontsize=20)
    axes[0, 0].xaxis.label.set_size(20)
    axes[0, 0].yaxis.label.set_size(20)


    # h, l = axes[0, 0].get_legend_handles_labels()
    # axes[0, 0].legend(handles=h, labels=['ES', 'HS'])
    # axes[0, 0].legend_.remove()
    # sns.histplot(x='MolWt', data=df, ax=axes[0, 0], hue='labels', kde=True, bins=20)
    sns.kdeplot(x='HBD', data=df, hue='labels', ax=axes[0, 1], linewidth=4, shade=True)
    axes[0, 1].legend(labels=['ES', 'HS'], fontsize=20)
    axes[0, 1].xaxis.label.set_size(20)
    axes[0, 1].yaxis.label.set_size(20)

    # sns.histplot(x='HBD', data=df, ax=axes[0, 1], hue='labels', kde=False, bins=20)
    sns.kdeplot(x='HBA', data=df, hue='labels', ax=axes[0, 2], linewidth=4, bw=0.3, shade=True)
    axes[0, 2].legend(labels=['ES', 'HS'], fontsize=20)
    axes[0, 2].xaxis.label.set_size(20)
    axes[0, 2].yaxis.label.set_size(20)


    sns.kdeplot(x='nRotB', data=df, hue='labels', ax=axes[1, 0], linewidth=4, bw=0.3, shade=True)
    axes[1, 0].legend(labels=['ES', 'HS'], fontsize=20)
    axes[1, 0].xaxis.label.set_size(20)
    axes[1, 0].yaxis.label.set_size(20)

    sns.kdeplot(x='LogP', data=df, hue='labels', ax=axes[1, 1], linewidth=4, shade=True, bw=0.3)
    axes[1, 1].legend(labels=['ES', 'HS'], fontsize=20)
    axes[1, 1].xaxis.label.set_size(20)
    axes[1, 1].yaxis.label.set_size(20)

    sns.kdeplot(x='TPSA', data=df, hue='labels', ax=axes[1, 2], linewidth=4, shade=True, bw=0.3)
    axes[1, 2].legend(labels=['ES', 'HS'], fontsize=20)
    axes[1, 2].xaxis.label.set_size(20)
    axes[1, 2].yaxis.label.set_size(20)

    fig.suptitle('Distribution map of ES and HS dataset (split by {} reaction path)'.format(threshold), fontsize=30)


    # fig.legend(labels=['ES', 'HS'], loc='upper center')


    # plt.legend()
    # plt.savefig('property_kdeplot_24w.png')
    plt.savefig(result)
    plt.show()

def sns_kde_plot(file):
    df = pd.read_csv(file)
    fig, axes = plt.subplots(2, 3, figsize=(20, 15))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot distribution picture of different data set')

    parser.add_argument('--in_file', type=str, help='Specify the absolute path to the property-file')
    parser.add_argument('--threshold', type=str, help='split criterion, can be 2, 3, 4')
    parser.add_argument('--out', type=str, help='Specify the absolute property path to the property picture')

    # args = parser.parse_args([
    #     '--in_file', '/data/baiqing/PycharmProjects/YaSAScore/data/cmpnn_data/24w_ES_HS_property.csv',
    #     '--out', '/data/baiqing/PycharmProjects/YaSAScore/data/cmpnn_data/24w_property_kdeplot.png'
    #
    # ])

    args = parser.parse_args()

    sns_kde(args.in_file, args.threshold, args.out)
    # sns_boxplot(property_file)