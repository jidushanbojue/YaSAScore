import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def stat_plot(file):
    sns.set_style('whitegrid')
    font_title = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 30}

    font_axis = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 20}


    df = pd.read_excel(file, sheet_name='Sheet3')
    fig = plt.figure(figsize=(20, 15))
    sns.barplot(x='Partition Criterion', y='AUC', data=df, hue='MODEL', saturation=0.3)
    sns.despine(top=True, right=True, left=False, bottom=False)

    plt.title('ROC AUC of different models', font_title)
    plt.xlabel('Partition Criterion', font_axis)
    plt.ylabel('ROC AUC', font_axis)
    plt.xticks(fontsize=15, weight='bold')
    plt.yticks(fontsize=15, weight='bold')
    plt.ylim(0.35, 0.85)
    plt.savefig('AUC_of_different_model.png', dpi=600)
    plt.show()
    print('Done')



if __name__ == '__main__':
    base_dir = '/data/baiqing/PycharmProjects/YaSAScore/picture'
    stat_file = os.path.join(base_dir, 'four_model_1.xls')

    stat_plot(stat_file)
