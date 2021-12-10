import os
import pandas as pd
import numpy as np
import random

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef, roc_curve, auc, confusion_matrix, f1_score

# y_pred = [0, 1, 0, 0]
# y_true = [0, 1, 1, 1]
# print('accuracy score:', accuracy_score(y_true=y_true, y_pred=y_pred))
# print('precision_score:', metrics.precision_score(y_true, y_pred))
# print('recall_score:', metrics.recall_score(y_true, y_pred))
# print('f1_score:', metrics.f1_score(y_true, y_pred))
# print('f0.5_score:', metrics.fbeta_score(y_true, y_pred, beta=0.5))
# print('f2_score:', metrics.fbeta_score(y_true, y_pred, beta=2.0))

def get_metrics_testset(src_file):
    df = pd.read_csv(src_file)

    y_true = df['ES_label']
    y_pred_count = df['class']
    y_pred_binary = df['class_binary']

    y_pred_syba = df['syba_class']
    y_pred_sa = df['sa_class']
    y_pred_sc = df['sc_class']
    

    
    matt_count = matthews_corrcoef(y_true, y_pred_count)
    matt_binary = matthews_corrcoef(y_true, y_pred_binary)
    matt_syba = matthews_corrcoef(y_true, y_pred_syba)
    matt_sa = matthews_corrcoef(y_true, y_pred_sa)
    matt_sc = matthews_corrcoef(y_true, y_pred_sc)
    
    accuray_count = accuracy_score(y_true, y_pred_count)
    accuracy_binary = accuracy_score(y_true, y_pred_binary)
    accuracy_syba = accuracy_score(y_true, y_pred_syba)
    accuracy_sa = accuracy_score(y_true, y_pred_sa)
    accuracy_sc = accuracy_score(y_true, y_pred_sc)

    fpr_count, tpr_count, threshold_count = roc_curve(y_true, df['ES_rate'])
    auc_count = auc(fpr_count, tpr_count)
    fpr_binary, tpr_binary, threshold_binary = roc_curve(y_true, df['ES_rate_binary'])
    auc_binary = auc(fpr_binary, tpr_binary)
    fpr_syba, tpr_syba, threshold_syba = roc_curve(y_true, df['syba_score'])
    auc_syba = auc(fpr_syba, tpr_syba)
    fpr_sa, tpr_sa, threshold_sa = roc_curve(y_true, df['sa_score'])
    auc_sa = auc(fpr_sa, tpr_sa)

    fpr_sc, tpr_sc, threshold_sc = roc_curve(y_true, df['sc_score'])
    auc_sc = auc(fpr_sc, tpr_sc)

    print('Done')

import matplotlib.pyplot as plt

from sklearn.utils import shuffle
def get_metrics_case(score_file):
    df = pd.read_csv(score_file)
    # df_source = pd.read_csv(source_file)


    # y_true = df['targets'] ### or df['p_np']
    y_true = df['p_np']

    y_pred_syba = df['syba_class']
    y_pred_mysyba = df['mysyba_class']
    y_pred_sa = df['sa_class']
    y_pred_sc = df['sc_class']
    y_pred_cmpnn = df['cmpnn_class']

    matt_syba = matthews_corrcoef(y_true, y_pred_syba)
    matt_syba_my = matthews_corrcoef(y_true, y_pred_mysyba)
    matt_sa = matthews_corrcoef(y_true, y_pred_sa)
    matt_sc = matthews_corrcoef(y_true, y_pred_sc)
    matt_cmpnn = matthews_corrcoef(y_true, y_pred_cmpnn)

    accuracy_syba = accuracy_score(y_true, y_pred_syba)
    accuracy_syba_my = accuracy_score(y_true, y_pred_mysyba)
    accuracy_sa = accuracy_score(y_true, y_pred_sa)
    accuracy_sc = accuracy_score(y_true, y_pred_sc)
    accuracy_cmpnn = accuracy_score(y_true, y_pred_cmpnn)


    fpr_syba, tpr_syba, threshold_syba = roc_curve(y_true, df['syba_score'])
    auc_syba = auc(fpr_syba, tpr_syba)
    fpr_syba_my, tpr_syba_my, threshold_syba_my = roc_curve(y_true, df['mysyba_score'])
    auc_syba_my = auc(fpr_syba_my, tpr_syba_my)
    fpr_sa, tpr_sa, threshold_sa = roc_curve(y_true, -df['sa_score'])
    auc_sa = auc(fpr_sa, tpr_sa)
    fpr_sc, tpr_sc, threshold_sc = roc_curve(y_true, -df['sc_score'])
    auc_sc = auc(fpr_sc, tpr_sc)
    fpr_cmpnn, tpr_cmpnn, threshold_cmpnn = roc_curve(y_true, df['cmpnn_score'])
    auc_cmpnn = auc(fpr_cmpnn, tpr_cmpnn)

    print('Done')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Generate metrics of different scoring function')
    parser.add_argument('--score_file', type=str, help='Specify the absolute path of score file')
    parser.add_argument('--source_file', type=str, help='Specify the source file')

    args = parser.parse_args([
        '--score_file', '/data/baiqing/PycharmProjects/YaSAScore/data/syba_data/80w_test_df_seed0_syba_and_mysyba_all_test_4_split_add_class.csv',
        # '--source_file', '/data/baiqing/PycharmProjects/YaSAScore/data/syba_data/24w_test_df_seed0.csv',
    ])

    # testset = os.path.join(base_dir, '24w_test_df_seed0_syba_and_mysyba_cmpnn_24w_test_add_class.csv')
    get_metrics_case(args.score_file)


