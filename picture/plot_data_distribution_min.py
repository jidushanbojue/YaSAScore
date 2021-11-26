import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def formatnum(x, pos):
    return '$%.1f$x$10^{4}$' % (x/10000)

formatter = FuncFormatter(formatnum)

def data_distribution_hist(file):
    df = pd.read_csv(file)
    # fig = plt.figure(figsize=(15, 10), dpi=400)
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    font_label = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 15}

    # sns.set_palette(sns.color_palette('bright'))
    sns.set(style='whitegrid')
    sns.countplot(x='min', data=df, color='lightskyblue')

    ax.yaxis.set_major_formatter(formatter)

    plt.axvline(x=2.5, ls='--', c='red')
    plt.axvline(x=1.5, ls='--', c='red')
    plt.axvline(x=3.5, ls='--', c='red')
    plt.xlabel('Minimum Reaction Steps', fontdict=font_label)
    plt.ylabel('Count', fontdict=font_label)
    plt.xticks(fontsize=15, weight='bold')
    plt.yticks(fontsize=15, weight='bold')
    plt.savefig('87w_min_distribution.png', dpi=600)
    plt.show()
    print('Done')


# all_data_87w = '/data/baiqing/src_data/reaction_step/article/87w.csv'
all_data_87w = '../data/87w.csv'
data_distribution_hist(all_data_87w)

