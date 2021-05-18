from nonpher import nonpher
from rdkit import Chem
import os
import pandas as pd
from multiprocessing import Pool


def worker(smi):
    # flag = False
    i = 0
    while True:
        morph = nonpher.complex_nonpher(smi, max_steps=1)
        if morph is not None:
            # flag = True
            return morph
        else:
            i = i + 1
            if i >= 10:
                return None
            continue
    # return morph

# def worker(smi):
#     morph = nonpher.complex_nonpher(smi)
#     return morph

def get_corresponding_nonpher_from_zinc(src_file, nonpher_file):
    df = pd.read_csv(src_file, names=['smiles', 'mol_name'])

    # for idx, smi in enumerate(df['smiles']):
    #     print(idx)
    #     flag = False
    #     while not flag:
    #         morph = nonpher.complex_nonpher(smi)
    #         if morph is not None:
    #             nonpher_result_list.append(morph)
    #             flag = True
    #         else:
    #             continue

    # nonpher_result_list = []
    # for idx, smi in enumerate(df['smiles']):
    #     print(idx, smi)
    #     morph = worker(smi)
    #     nonpher_result_list.append(morph)

    p = Pool(160)
    result = p.map(worker, df['smiles'])
    p.close()

    # df['nonpher'] = nonpher_result_list
    df['nonpher'] = result

    df.to_csv(nonpher_file)




if __name__ == "__main__":
    base_dir = '/data/baiqing/src_data/reaction_step/case_study/zinc_Nonpher'
    zinc_file = os.path.join(base_dir, 'Zinc_top10000.csv')
    nonpher_file = os.path.join(base_dir, 'Zinc_top10000_nonpher_maxsteps1.csv')
    get_corresponding_nonpher_from_zinc(zinc_file, nonpher_file)


