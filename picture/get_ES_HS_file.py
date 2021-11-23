import os
import numpy as np
import pandas as pd

def get_ES_HS_file(train_file, val_file, ES_train_result, HS_train_result):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    df = pd.concat([train_df, val_df])
    ES = df[df['targets']==1]
    HS = df[df['targets']==0]

    ES.to_csv(ES_train_result)
    HS.to_csv(HS_train_result)
    print('Done!!!')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot distribution picture of different data set')
    # base_dir = 'projects/data'

    parser.add_argument('--train_file', type=str, help='Specify the absolute path to the ES-file')
    parser.add_argument('--val_file', type=str, help='Specify the absolute path to the HS-file')
    parser.add_argument('--ES_out', type=str, help='Specify the absolute property path to the picture')
    parser.add_argument('--HS_out', type=str, help='Specify the absolute property path to the ')

    # args = parser.parse_args([
    #     '--train_file', '/data/baiqing/PycharmProjects/YaSAScore/data/cmpnn_data/24w_train_df_seed0.csv',
    #     '--val_file', '/data/baiqing/PycharmProjects/YaSAScore/data/cmpnn_data/24w_val_df_seed0.csv',
    #     '--ES_out', '/data/baiqing/PycharmProjects/YaSAScore/data/cmpnn_data/24w_ES.csv',
    #     '--HS_out', '/data/baiqing/PycharmProjects/YaSAScore/data/cmpnn_data/24w_HS.csv'
    #
    # ])

    args = parser.parse_args()

    get_ES_HS_file(args.train_file, args.val_file, args.ES_out, args.HS_out)

