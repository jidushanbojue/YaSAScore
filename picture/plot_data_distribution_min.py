import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def data_distribution_hist(file):
    df = pd.read_csv(file)
    fig = plt.figure(figsize=(15, 10), dpi=400)
    sns.set_palette(sns.color_palette('bright'))
    sns.countplot(x='min', data=df)
    # plt.his(df['min'])
    plt.axvline(x=2.5, ls='--', c='red')
    plt.xlabel('Minimum Reaction Steps', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('87w_min_distribution.png')
    plt.show()
    print('Done')


all_data_87w = '/data/baiqing/src_data/reaction_step/article/87w.csv'
data_distribution_hist(all_data_87w)

