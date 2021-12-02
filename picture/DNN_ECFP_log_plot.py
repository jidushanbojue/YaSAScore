import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_log(log_file, result_picture):
    df = pd.read_csv(log_file)

    fig = plt.figure(figsize=(20, 15), dpi=600)
    sns.lineplot(x='epoch', y='loss', data=df, label='Train Loss', linewidth=5, estimator=None)
    sns.lineplot(x='epoch', y='val_loss', data=df, label='Validate Loss', linewidth=5, color='red', estimator=None)

    plt.legend()
    plt.title("Loss of DNN-ECFP model", fontsize=30)
    plt.tick_params(labelsize=30)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.savefig(result_picture)
    # plt.show()
    print('Done')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot log file')
    # base_dir = 'projects/data'

    parser.add_argument('--in_file', type=str, help='Specify the absolute path to log file of model')
    parser.add_argument('--out', type=str, help='Specify the absolute path of log picture')

    args = parser.parse_args([
        '--in_file', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/split_by_4/split_4.log',
        '--out', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/split_by_4/split_4_log.png'
    ])

    # base_dir = './'
    # log_file_DNN_ecfp = os.path.join('/home/baiqing/DNN_ECFP', '24w_training_last_log_ecfp_3_3_0.0001_3_layers_1.log')
    # log_file_DNN_six_descriptor = os.path.join('/home/baiqing/', '24w_training_last_log_six_descriptor_3_3_0.0001_3_layers_1.log')
    # plot_log(log_file_DNN_ecfp)
    plot_log(args.in_file, args.out)