import os
import pandas as pd
import sys
from amol_utils_rdchiral import Reaction, Parsers

datafile = '/home/cadd/Data/pistachio/temp/temp100.csv'

p = Parsers()
print('Parsing: {}'.format(datafile))
reaction_data = p.import_USPTO(datafile)

dataset = pd.DataFrame(columns=['ID', 'reaction_hash', 'reactants', 'products', 'classification', 'retro_template', 'template_hash', 'selectivity', 'outcomes'])
failed = []


rxn_number = 0
extracted = 0

for index, line in reaction_data.iterrows():
    rxn_number += 1
    print(index)
    print('Reaction: {}'.format(rxn_number))
    reaction = Reaction(line['rsmi'], rid=line['ID'])
    # try:
    #     if len(reaction.rsmi.split('>')) > 3:
    #         failed.append([line['ID'], line['rsmi'], 'compoments > 3'])
    #         continue
    #     elif len(reaction.product_list) > 1:
    #         failed.append(line['ID'], line['rsmi'], 'products > 1')
    #         continue
    #     elif reaction.incomplete_reaction():
    #         failed.append([line['ID'], line['rsmi'], 'incomplete'])
    #         continue
    #     elif reaction.equivalent_reactant_product_set():
    #         failed.append([line['ID'], line['rsmi'], 'reactants = products'])
    #         continue
    #     elif reaction.generate_reaction_template(radius=1) is None:
    #         failed.append([line['ID'], line['rsmi'], 'template generation failure'])
    #         continue
    #     elif reaction.validate_retro_template(reaction.retro_template) is None:
    #         failed.append([line['ID'], line['rsmi'], 'template rdkit validation failed'])
    #         continue
    #     elif reaction.check_retro_template_outcome(reaction.retro_template, reaction.products, save_outcome=True) != 0:
    #         outcomes = len(reaction.retro_outcomes)
    #         assessment = reaction.assess_retro_template(reaction.retro_template, reaction.reactant_mol_list, reaction.retro_outcome)
    #     else:
    #         pass
    # except :
    #     continue

    try:
        if len(reaction.rsmi.split('>')) > 3:
            failed.append([line["ID"], line["rsmi"], "components > 3"])
            continue
        elif len(reaction.product_list) > 1:
            failed.append([line["ID"], line["rsmi"], "products > 1"])
            continue
        elif reaction.incomplete_reaction():
            failed.append([line["ID"], line["rsmi"], "incomplete"])
            continue
        elif reaction.equivalent_reactant_product_set():
            failed.append([line["ID"], line["rsmi"], "reactants = products"])
            continue
        elif reaction.generate_reaction_template(radius=1) is None:
            failed.append([line["ID"], line["rsmi"], "template generation failure"])
            continue
        elif reaction.validate_retro_template(reaction.retro_template) is None:
            failed.append([line["ID"], line["rsmi"], "template rdkit validation failed"])
            continue
        elif reaction.check_retro_template_outcome(reaction.retro_template, reaction.products, save_outcome=True) != 0:
            outcomes = len(reaction.retro_outcomes)
            assessment = reaction.assess_retro_template(reaction.retro_template, reaction.reactant_mol_list, reaction.retro_outcomes)
        else:
            pass
    except:
        continue


    rinchi_hash = reaction.generate_concatenatedRInChI()
    line_list = [
        line['ID'],
        rinchi_hash,
        reaction.reactants,
        reaction.products,
        line['classification'],
        reaction.retro_template,
        reaction.hash_template(reaction.retro_template),
        assessment,
        outcomes
    ]

    line = pd.DataFrame([line_list], columns=["ID", "reaction_hash", "reactants", "products", "classification", "retro_template", "template_hash", "selectivity", "outcomes"])
    dataset = dataset.append(line, sort=False)
    extracted += 1
    sys.stdout.flush()

print('Extracted: {}'.format(extracted))
failed_df = pd.DataFrame(failed, columns=['ID', 'rsmi', 'reason'])
output = dataset.drop_duplicates(subset='reaction_hash')
print(output)
print(failed_df)

