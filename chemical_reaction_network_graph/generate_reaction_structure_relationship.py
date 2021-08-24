import pandas as pd
import os
from multiprocessing import Pool
from functools import partial
import argparse
from contextlib import closing
from joblib import Parallel, delayed

def process(reaction_file=None, structure_file=None, result_folder=None):

    dir, filename = os.path.split(reaction_file)
    print(filename)
    # result_file = os.path.join(dir, filename+'_res.csv')
    result_file = os.path.join(result_folder, filename)
    df_reaction = pd.read_csv(reaction_file, names=['ID', 'idx', 'reagents', 'new_reactants', 'product_inchi'])
    df_structure = pd.read_csv(structure_file, names=['structure', 'idx'])

    res_df = pd.DataFrame()
    for idx, line in df_reaction.iterrows():
        print(idx)

        # reagent_list = [el.strip() for el in line['reagents'].split('||') if el.startswith('InChI')]

        try:
            reagent_list = [el.strip() for el in line['reagents'].split('||') if el.startswith('InChI')]
        except AttributeError:
            reagent_list = []
        try:

            reactant_list = [el.strip() for el in line['new_reactants'].split('||') if el.startswith('InChI')]
        except AttributeError:
            reactant_list = []
        products_list = [el.strip() for el in line['product_inchi'].split('||') if el.startswith('InChI')]
        structure_list = reagent_list + reactant_list + products_list
        structure_src_df = pd.DataFrame({'structure': structure_list})
        role_list = ['Reagent'] * len(reagent_list) + ['Reactant'] * len(reactant_list) + ['Product'] * len(products_list)

        structure_filter_df = df_structure[df_structure['structure'].map(lambda x: x in structure_list)]
        structure_merge_df = pd.merge(structure_src_df, structure_filter_df)
        structure_id_list = list(structure_merge_df['idx'])
        reaction_id_list = [line['idx']] * len(structure_list)
        index = [str(line['idx']) + '_' + str(i) for i in range(len(structure_list))]
        dic = {'role': role_list, 'structure_id': structure_id_list, 'reaction_id': reaction_id_list}
        temp_df = pd.DataFrame(data=dic, index=index)
        # try:
        #     temp_df = pd.DataFrame(data=dic, index=index)
        # except ValueError:
        #     print('failed: {}'.format(filename))
        #     temp_df = pd.DataFrame()
        res_df = res_df.append(temp_df)
    res_df.to_csv(result_file)

        # for reagent in reagent_list:
            # print(reagent)
            # if reagent in structure_series:
            #     print('True')





if __name__ == '__main__':
    # base_dir = '/home/baiqing/src_data/reaction_step/uspto_pistachio'
    # # reaction_file = os.path.join(base_dir, 'all_reaction_746.csv')
    # structure_file = os.path.join(base_dir, 'all_structure.csv')
    # # result_file = os.path.join(base_dir, 'test.csv')
    # # worker(reaction_file, structure_file, result_file=result_file)
    #
    # reaction_file_list = [os.path.join(base_dir, 'reaction_split', filename) for filename in os.listdir(os.path.join(base_dir, 'reaction_split'))]
    # worker = partial(process, structure_file=structure_file)
    # # for file in reaction_file_list:
    # #     worker(file)
    #
    # pool = Pool(50)
    # pool.map(worker, reaction_file_list)

    parser = argparse.ArgumentParser(description='Process subfile reaction')
    parser.add_argument('-d', '--data', type=str, default=None, help='Specify the absolute path to the reaction file folder')
    parser.add_argument('-o', '--out', type=str, default=None, help='Specify the result file folder')
    parser.add_argument('-s', '--structure', type=str, default=None, help='Specify the structure file')
    # parser.
    # args = parser.parse_args(['-d', '/home/baiqing/src_data/reaction_step/uspto_pistachio/test', '-o', '/home/baiqing/src_data/reaction_step/uspto_pistachio/test_result', '-s', '/home/baiqing/src_data/reaction_step/uspto_pistachio/all_structure.csv'])
    # args = parser.parse_args(['-d', '/data/baiqing/src_data/reaction_step/uspto/uspto_reaction_split', '-o', '/data/baiqing/src_data/reaction_step/uspto/uspto_reaction_split_result', '-s', '/data/baiqing/src_data/reaction_step/uspto/all_structure_uspto.csv'])
    args = parser.parse_args()
    data_source = args.data

    if os.path.exists(args.out):
        pass
    else:
        os.mkdir(args.out)

    reaction_file_list = [os.path.join(data_source, filename) for filename in os.listdir(data_source)]

    worker = partial(process, structure_file=args.structure, result_folder=args.out)

    # for file in reaction_file_list:
    #     worker(file)


    Parallel(n_jobs=10)(delayed(worker)(reaction_file) for reaction_file in reaction_file_list)

    # with closing(Pool(50)) as pool:
    # pool = Pool(50)
    #     pool.map(worker, reaction_file_list)


