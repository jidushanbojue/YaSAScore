import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
import gzip
import os

from syba.syba import SybaClassifier, SmiMolSupplier
from scscore.standalone_model_numpy import SCScorer
import sascorer as sa
from sklearn.ensemble import RandomForestClassifier
from nonpher import complex_lib as cmplx
import pandas as pd
from multiprocessing import Pool
from functools import partial
from joblib import Parallel, delayed

def worker(syba_model, my_syba_model,scscore_model, smi):
    try:
        syba_score = syba_model.predict(smi)
    except:
        syba_score = 0
    try:
        my_syba_score = my_syba_model.predict(smi)
    except :
        my_syba_score = 0
    sa_score = sa.calculateScore(Chem.MolFromSmiles(smi))
    sc_score = scscore_model.get_score_from_smi(smi)[1]
    return syba_score, my_syba_score, sa_score, sc_score     
   

def generate_compare_result(src_file, result_file):
    df = pd.read_csv(src_file)
    syba = SybaClassifier()
    syba.fitDefaultScore()
    syba_my = SybaClassifier()
    syba_my.fitDefaultScore_my()
    scscore = SCScorer()
    scscore.restore()
    partial_worker = partial(worker, syba, syba_my, scscore)

    # result_list = []
    syba_list = []
    my_syba_list = []
    sa_list = []
    sc_list = []
    for idx, smi in enumerate(df['smiles']):
        print(idx)
        # if idx <=1056:
        #     continue
        # result = worker(syba_model=syba, scscore_model=scscore, smi=smi)
        res = partial_worker(smi=smi)
        syba_list.append(res[0])
        my_syba_list.append(res[1])
        sa_list.append(res[2])
        sc_list.append(res[3])
        # result_list.append(res)


    # print('Begining to get three models result')
    # pool = Pool(28)
    # result = pool.map(partial_worker, df['smiles'][:100])
    # result_df = pd.DataFrame(result_list, columns=['syba_score', 'sa_score', 'sc_score'])
    df['syba_score'] = syba_list
    df['mysyba_score'] = my_syba_list
    df['sa_score'] = sa_list
    df['sc_score'] = sc_list
    df.to_csv(result_file)
    print('Done!')
   
        
if __name__ == '__main__':
    # base_dir = '/home/cadd/PycharmProjects/sascore_scscore_syba_syba2_model/my_data'
    # src_file = os.path.join(base_dir, 'all_structure_smiles.csv')
    # result_file = os.path.join(base_dir, 'three_models_result.csv')
    # generate_compare_result(src_file=src_file, result_file=result_file)

    base_dir = '/home/baiqing/PycharmProjects/syba-master/my_data'
    # src_file = os.path.join(base_dir, 'top50_count_binary_merged_test_min_3_3_DNN.csv')
    # result_file = os.path.join(base_dir, 'testset_four_models_result_DNN_new_syba.csv')

    # src_file = os.path.join(base_dir, 'top50_count_binary_merged_case3_min_3_3_DNN.csv')
    # result_file = os.path.join(base_dir, 'case3_four_models_result_new_syba.csv')

    # src_file = os.path.join(base_dir, 'top50_count_binary_merged_case1_min_3_3_DNN.csv')
    # result_file = os.path.join(base_dir, 'case1_four_models_result.csv')

    # src_file = os.path.join(base_dir, 'top50_count_binary_merged_alltest_min_3_3_DNN.csv')
    # result_file = os.path.join(base_dir, 'testset_four_models_result_DNN_syba_and_mysyba.csv')

    # src_file = os.path.join(base_dir, 'steps_add_degree_new_24w_cluster.csv')
    # result_file = os.path.join(base_dir, '24w_cluster_syba_and_mysyba.csv')

    # src_file = os.path.join(base_dir, '24w_test_df_seed0.csv')
    # result_file = os.path.join(base_dir, '24w_test_df_seed0_syba_and_mysyba.csv')

    src_file = os.path.join(base_dir, '24w_cmpnn_remain_all_test.csv')
    result_file = os.path.join(base_dir, '24w_test_df_seed0_syba_and_mysyba_all_test.csv')


    generate_compare_result(src_file=src_file, result_file=result_file)
    
    
    