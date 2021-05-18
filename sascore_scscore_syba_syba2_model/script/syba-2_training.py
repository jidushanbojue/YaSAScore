# from rdkit import Chem
# from syba.syba import SybaClassifier
#
# syba = SybaClassifier()
# syba.fitDefaultScore()
# smi = 'O=C(C)Oc1ccccc1C(=O)O'
#
# print(syba.predict(smi))

import math
import gzip
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from IPython.display import SVG
from syba import syba
from itertools import islice
import pandas as pd
from collections import defaultdict

def processFile(path, smi_col=1):
    """
    For given compressed file, function returns number of all compounds and all Morgan fragment types with corresponding number of compounds
    in which they are found.

    path to data set (gzip csv file)
    smi_col is column where are SMILES, first column has index 0!

    return: dictionary with fragments and their counts and the number of processed compounds as tuple
    """
    # with gzip.open(path, mode="rt") as reader:
    with open(path, mode='rt') as reader:
        suppl = syba.SmiMolSupplier(reader, header=True, smi_col=smi_col) # reads reader line by line and returns tuple with RDMol and splitted line
        fs = {}
        n_of_compounds = 0

        for m, *spls in suppl:
            try:
                for frag in Chem.GetMorganFingerprint(m,2).GetNonzeroElements().keys():
                    d = fs.setdefault(frag, [0])
                    d[0] += 1
                n_of_compounds += 1
            except Exception as e:
                print(spls)
                raise e
        return fs, n_of_compounds

def processFile_1(path):
    df = pd.read_csv(path)
    fs = defaultdict(int)

    for idx, line in df.iterrows():
        print(idx)
        mol = Chem.MolFromSmiles(line['smiles'])
        for frag in Chem.GetMorganFingerprint(mol, 2).GetNonzeroElements().keys():
            fs[frag] += 1

    return fs, len(df)



hs_fragments, hs_compounds = processFile_1('data/24w_train_HS.csv')
es_fragments, es_compounds = processFile_1('data/24w_train_ES.csv')

#
def mergeFragmentCounts(es_fragments, hs_fragments):
    fragments = set(es_fragments.keys())
    fragments.update(hs_fragments.keys())
    fragment_counts = {}

    for f in fragments:
        fragment_counts[f] = (es_fragments.get(f, 0), hs_fragments.get(f, 0))
    return fragment_counts
#
fragment_counts = mergeFragmentCounts(es_fragments, hs_fragments)
syba.writeCountFile('syba_ES_cluster_HS_train_val.csv', fragment_counts, (es_compounds, hs_compounds))
syba.writeScoreFile('syba_ES_cluster_HS_score_train_val.csv', fragment_counts, (es_compounds, hs_compounds))
print('Done!!!')



