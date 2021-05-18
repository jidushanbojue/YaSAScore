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
    base_dir = '~/PycharmProjects/CMPNN-master/data'
    train_file = os.path.join(base_dir, '24w_train_df_seed0.csv')
    val_file = os.path.join(base_dir, '24w_val_df_seed0.csv')

    ES_train_seed0 = os.path.join(base_dir, '24w_train_ES.csv')
    HS_train_seed0 = os.path.join(base_dir, '24w_train_HS.csv')

    get_ES_HS_file(train_file, val_file, ES_train_seed0, HS_train_seed0)
