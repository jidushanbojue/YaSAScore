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
def get_metrics_case(src_file):
    # random.seed(0)
    sizes = (0.8, 0.1, 0.1)

    df = pd.read_csv(src_file)
    # data_len = int(len(df)/2)

    # y_true_list = [1] * data_len + [0] * data_len
    # df['y_true'] = y_true_list
    # # y_true = df['y_true']
    #
    # df = shuffle(df, random_state=0)
    # train_size = int(sizes[0] * len(df))
    # train_val_size = int((sizes[0] +  sizes[1]) * len(df))
    #
    # train = df[:train_size]
    # val = df[train_size: train_val_size]
    # test = df[train_val_size:]

    y_true = df['targets']

    # y_pred_count = df['y_pred']
    # y_pred_binary = df['y_pred_binary']
    # y_pred_count = df['class']
    # y_pred_binary = df['class_binary']
    #
    y_pred_syba = df['syba_class']
    y_pred_mysyba = df['syba_class_my']
    y_pred_sa = df['sa_class']
    y_pred_sc = df['sc_class']
    y_pred_cmpnn = df['cmpnn_class']
    # count_matrix = confusion_matrix(y_true, y_pred_count).ravel()
    # (tn, fp, fn, tp) = count_matrix
    # f1 = f1_score(y_true, y_pred_count)
    # matt_count = matthews_corrcoef(y_true, y_pred_count)
    # matt_binary = matthews_corrcoef(y_true, y_pred_binary)
    matt_syba = matthews_corrcoef(y_true, y_pred_syba)
    matt_syba_my = matthews_corrcoef(y_true, y_pred_mysyba)
    matt_sa = matthews_corrcoef(y_true, y_pred_sa)
    matt_sc = matthews_corrcoef(y_true, y_pred_sc)
    matt_cmpnn = matthews_corrcoef(y_true, y_pred_cmpnn)

    # accuray_count = accuracy_score(y_true, y_pred_count)
    # accuracy_binary = accuracy_score(y_true, y_pred_binary)
    accuracy_syba = accuracy_score(y_true, y_pred_syba)
    accuracy_syba_my = accuracy_score(y_true, y_pred_mysyba)
    accuracy_sa = accuracy_score(y_true, y_pred_sa)
    accuracy_sc = accuracy_score(y_true, y_pred_sc)
    accuracy_cmpnn = accuracy_score(y_true, y_pred_cmpnn)

    # fpr_count, tpr_count, threshold_count = roc_curve(y_true, df['ES_rate'])
    # auc_count = auc(fpr_count, tpr_count)
    # fpr_binary, tpr_binary, threshold_binary = roc_curve(y_true, df['ES_rate_binary'])
    # auc_binary = auc(fpr_binary, tpr_binary)

    # fpr_count, tpr_count, threshold_count = roc_curve(y_true, df['y_prob'])
    # auc_count = auc(fpr_count, tpr_count)
    # fpr_binary, tpr_binary, threshold_binary = roc_curve(y_true, df['y_prob_binary'])
    # auc_binary = auc(fpr_binary, tpr_binary)

    fpr_syba, tpr_syba, threshold_syba = roc_curve(df['targets'], df['syba_score'])
    auc_syba = auc(fpr_syba, tpr_syba)
    fpr_syba_my, tpr_syba_my, threshold_syba_my = roc_curve(df['targets'], df['mysyba_score'])
    auc_syba_my = auc(fpr_syba_my, tpr_syba_my)
    fpr_sa, tpr_sa, threshold_sa = roc_curve(df['targets'], -df['sa_score'])
    auc_sa = auc(fpr_sa, tpr_sa)
    fpr_sc, tpr_sc, threshold_sc = roc_curve(df['targets'], -df['sc_score'])
    auc_sc = auc(fpr_sc, tpr_sc)
    fpr_cmpnn, tpr_cmpnn, threshold_cmpnn = roc_curve(df['targets'], df['cmpnn_score'])
    auc_cmpnn = auc(fpr_cmpnn, tpr_cmpnn)

    print('Done')


if __name__ == '__main__':
    base_dir = '/home/baiqing/PycharmProjects/syba-master/my_data'
    # testset = os.path.join(base_dir, 'testset_four_models_result_add_class.csv')
    # testset = os.path.join(base_dir, 'testset_four_models_result_add_class_new_syba.csv')
    # testset = os.path.join(base_dir, 'alltestset_four_models_result_add_class_syba_mysyba.csv')

    case2 = os.path.join(base_dir, 'case2_four_models_result_add_class_RF.csv')
    case3 = os.path.join(base_dir, 'case3_four_models_result_add_class.csv')
    case3 = os.path.join(base_dir, 'case3_four_models_result_add_class_new_syba.csv')

    # testset = os.path.join(base_dir, '24w_cluster_syba_and_mysyba.csv')
    # testset = os.path.join(base_dir, '24w_test_df_seed0_syba_and_mysyba_add_class.csv')
    # testset = os.path.join(base_dir, '24w_test_df_seed0_syba_and_mysyba_all_test_add_class.csv')
    # testset = os.path.join(base_dir, '24w_test_df_seed0_syba_and_mysyba_cmpnn_all_test_add_class.csv')
    testset = os.path.join(base_dir, '24w_test_df_seed0_syba_and_mysyba_cmpnn_24w_test_add_class.csv')
    get_metrics_case(testset)
    # get_metrics_testset(testset)


