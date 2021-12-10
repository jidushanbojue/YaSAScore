import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Generate metrics of different scoring function')
parser.add_argument('--input_file', type=str, help='Specify the absolute path of score file')
parser.add_argument('--cmpnn_input_file', type=str, help='Specify the absolute path of cmpnn predict file')
parser.add_argument('--out', type=str, help='Specify the source file')
# args = parser.parse_args([
#     '--input_file', '/data/baiqing/PycharmProjects/YaSAScore/data/syba_data/24w_test_df_seed0_syba_and_mysyba.csv',
#     '--out', '/data/baiqing/PycharmProjects/YaSAScore/data/syba_data/24w_test_df_seed0_syba_and_mysyba_add_class.csv'
# ])

args = parser.parse_args()


df = pd.read_csv(args.input_file)


syba_class = []
my_syba_class = []
sa_class = []
sc_class = []
for idx, line in df.iterrows():
    if line['syba_score'] >=0:  ### or -18.6
        syba_class.append(1)
    else:
        syba_class.append(0)

    if line['sa_score'] >= 6:   ### or 4.5
        sa_class.append(0)
    else:
        sa_class.append(1)

    if line['sc_score'] >= 2.5:
        sc_class.append(0)
    else:
        sc_class.append(1)

    if line['mysyba_score'] >=0:
        my_syba_class.append(1)
    else:
        my_syba_class.append(0)

df['syba_class'] = syba_class
df['mysyba_class'] = my_syba_class
df['sa_class'] = sa_class
df['sc_class'] = sc_class

cmpnn_class = []
cmpnn_score = []

df_cmpnn = pd.read_csv(args.cmpnn_input_file)
for idx, line in df_cmpnn.iterrows():

    if line['pred_0'] >= 0.5:

        cmpnn_class.append(1)

    else:
        cmpnn_class.append(0)

df['cmpnn_score'] = df_cmpnn['pred_0']
df['cmpnn_class'] = cmpnn_class


df.to_csv(args.out)
