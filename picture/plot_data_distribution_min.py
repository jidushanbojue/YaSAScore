import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def formatnum_10000(x, pos):
    return '$%.1f$x$10^{4}$' % (x/10000)

def formatnum_1000(x, pos):
    return '$%.1f$x$10^{3}$' % (x/1000)

def formatnum_100(x, pos):
    return '$%.1f$x$10^{2}$' % (x/100)

formatter_10000 = FuncFormatter(formatnum_10000)
formatter_1000 = FuncFormatter(formatnum_1000)
formatter_100 = FuncFormatter(formatnum_100)

def data_distribution_hist(file):
    df = pd.read_csv(file)

    fig = plt.figure(figsize=(15, 10))
    # fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    font_label = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 20}


    sns.set(style='darkgrid')
    ax = sns.countplot(x='min', data=df, color='lightskyblue')


    ax.yaxis.set_major_formatter(formatter_10000)

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

def data_distribution_hist_small_threshold_7_11(file):
    df = pd.read_csv(file)

    df_small = df[df['min']>=7]

    fig = plt.figure(figsize=(15, 10))
    # fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    font_label = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 20}


    sns.set(style='darkgrid')
    ax = sns.countplot(x='min', data=df_small, color='lightskyblue')


    ax.yaxis.set_major_formatter(formatter_1000)


    # plt.axvline(x=2.5, ls='--', c='red')
    # plt.axvline(x=1.5, ls='--', c='red')
    # plt.axvline(x=3.5, ls='--', c='red')
    plt.xlabel('Minimum Reaction Steps', fontdict=font_label)
    plt.ylabel('Count', fontdict=font_label)
    plt.xticks(fontsize=15, weight='bold')
    plt.yticks(fontsize=15, weight='bold')
    plt.savefig('87w_min_distribution_7_11.png', dpi=600)
    plt.show()
    print('Done')

def data_distribution_hist_small_threshold_12_17(file):
    df = pd.read_csv(file)

    df_small = df[df['min']>=12]

    fig = plt.figure(figsize=(15, 10))
    # fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    font_label = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 20}


    sns.set(style='darkgrid')
    ax = sns.countplot(x='min', data=df_small, color='lightskyblue')


    # ax.yaxis.set_major_formatter(formatter_100)


    # plt.axvline(x=2.5, ls='--', c='red')
    # plt.axvline(x=1.5, ls='--', c='red')
    # plt.axvline(x=3.5, ls='--', c='red')
    plt.xlabel('Minimum Reaction Steps', fontdict=font_label)
    plt.ylabel('Count', fontdict=font_label)
    plt.xticks(fontsize=15, weight='bold')
    plt.yticks(fontsize=15, weight='bold')
    plt.savefig('87w_min_distribution_12.png', dpi=600)
    plt.show()
    print('Done')

# all_data_87w = '/data/baiqing/src_data/reaction_step/article/87w.csv'
all_data_87w = '../data/87w.csv'
data_distribution_hist(all_data_87w)
data_distribution_hist_small_threshold_7_11(all_data_87w)
data_distribution_hist_small_threshold_12_17(all_data_87w)


