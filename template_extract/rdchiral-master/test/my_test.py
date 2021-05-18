# import sys, os
# sys.path = [os.path.dirname(os.getcwd())] + sys.path
#
# print(sys.path)
#
# import rdkit
# import rdkit.Chem as Chem
# import rdkit.Chem.AllChem as AllChem
# print(rdkit.rdBase.rdkitVersion)
#
# from IPython.display import display, Image
# import rdchiral
# print(rdchiral.__path__)
# from rdchiral.main import rdchiralRun, rdchiralRunText, rdchiralReactants, rdchiralReaction


# def sep_bar():
#     print('')
#     for i in range(3):
#         print('=' * 80)
#     print('')
#
# reaction_smarts = '[C:1][OH:2]>>[C:1][O:2][C]'
# reactant_smiles = 'CC(=O)OCCCO'
#
# def show_outcomes(reaction_smarts, reactant_smiles):
#     outcomes_rdkit_mol = AllChem.ReactionFromSmarts(reaction_smarts).RunReactants((Chem.MolFromSmiles(reactant_smiles), ))
#     outcomes_rdkit = set()
#     for outcome in outcomes_rdkit_mol:
#         outcomes_rdkit.add('.'.join(sorted([Chem.MolToSmiles(x) for x in outcome])))
#
#     outcomes_rdchiral = rdchiralRunText(reaction_smarts, reactant_smiles)
#     print('Reaction SMARTS: {}'.format(reaction_smarts))
#     display(Chem.MolFromSmiles(reactant_smiles))
#     print('Input SMILES: {}'.format(reactant_smiles))
#
#     if outcomes_rdkit:
#         display(Chem.MolFromSmiles('.'.join(outcomes_rdkit)))
#     print('{:1d} RDKit outcomes: {}'.format(len(outcomes_rdkit), '.'.join(outcomes_rdkit)))
#
#     if outcomes_rdchiral:
#         display(Chem.MolFromSmiles('.'.join(outcomes_rdchiral)))
#     print('{:1d} RDChiral outcomes: {}'.format(len(outcomes_rdchiral), '.'.join(outcomes_rdchiral)))
#
# show_outcomes(reaction_smarts, reactant_smiles)

# rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OD1].[N!H0:3]>>[C:1](=[O:2])[N:3]')
# print(rxn.GetNumProductTemplates())
#
# ps = rxn.RunReactants((Chem.MolFromSmiles('CC(=O)O'),Chem.MolFromSmiles('NC')))
#
#
# rxn = AllChem.ReactionFromMolecule()
# print('Done')


from rdkit.Chem import Draw
from rdkit.Chem import AllChem
# rxn = AllChem.ReactionFromSmarts('[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])O.[N-:13]=[N+:14]=[N-:15]>C(Cl)Cl.C(=O)(C(=O)Cl)Cl>[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])[N:13]=[N+:14]=[N-:15]',useSmiles=True)
# d2d = Draw.MolDraw2DCairo(800, 300)
# d2d.DrawReaction(rxn)
# png = d2d.GetDrawingText()
# open('reaction1.png', 'wb+').write(png)
#
# d2d = Draw.MolDraw2DCairo(800, 300)
# d2d.DrawReaction(rxn, highlightByReactant=True)
# png = d2d.GetDrawingText()
# open('reaction2.png', 'wb+').write(png)

# rxn = AllChem.ReactionFromSmarts('([I;H0;D1;+0]-[c;H0;D3;+0:2](:[c:1]):[c:3])>>([c:1]:[cH;D2;+0:2]:[c:3])')
# d2d = Draw.MolDraw2DCairo(800, 300)
# d2d.DrawReaction(rxn, highlightByReactant=True)
# png = d2d.GetDrawingText()
# open('reaction3.png', 'wb+').write(png)


# rxn = AllChem.ReactionFromSmarts('CCO.CN.II.O.[CH3:1][O:2][CH2:3][O:4][c:5]1[cH:6][cH:7][c:8]([CH2:9][c:10]2[cH:11][c:12]([CH3:13])[c:14]([OH:15])[cH:16][c:17]2[CH3:18])[cH:19][c:20]1[CH:21]([CH3:22])[CH3:23]>>I[c:16]1[c:14]([OH:15])[c:12]([CH3:13])[cH:11][c:10]([CH2:9][c:8]2[cH:7][cH:6][c:5]([O:4][CH2:3][O:2][CH3:1])[c:20]([CH:21]([CH3:22])[CH3:23])[cH:19]2)[c:17]1[CH3:18]')
# d2d = Draw.MolDraw2DCairo(800, 300)
# d2d.DrawReaction(rxn)
# png = d2d.GetDrawingText()
# open('reaction3.png', 'wb+').write(png)

# rxn = AllChem.ReactionFromSmarts('C[O:1][c:2]1[cH:3][c:4]([c:5]([cH:6][c:7]1[Cl:8])[CH:9]=[O:10])[Cl:11]>CN(C)C=O.Cl[Li]>[O:10]=[CH:9][c:5]1[cH:6][c:7]([c:2]([cH:3][c:4]1[Cl:11])[OH:1])[Cl:8]')
# d2d = Draw.MolDraw2DCairo(800, 300)
# d2d.DrawReaction(rxn, highlightByReactant=True)
# png = d2d.GetDrawingText()
# open('reaction3.png', 'wb+').write(png)

rxn = AllChem.ReactionFromSmarts('[CH3:1][CH2:2][O:3][C:4](=[O:5])[c:6]1[cH:7][c:8]([cH:9][c:10]([cH:11]1)[C:12](=[O:13])O)-[c:14]1[cH:15][cH:16][c:17]([cH:18][c:19]1[C:20]#[N:21])[CH3:22].[OH:23][CH:24]1[CH2:25][NH:26][CH2:27]1.Cl>CCN=C=NCCCN(C)C.On1nnc2ccccc21.CC(C)N(CC)C(C)C.ClCCl.O.Cl>[CH3:1][CH2:2][O:3][C:4](=[O:5])[c:6]1[cH:7][c:8]([cH:9][c:10]([cH:11]1)[C:12](=[O:13])[N:26]1[CH2:27][CH:24]([CH2:25]1)[OH:23])-[c:14]1[cH:15][cH:16][c:17]([cH:18][c:19]1[C:20]#[N:21])[CH3:22]')
# d2d = Draw.MolDraw2DCairo(800, 300)
# d2d.DrawReaction(rxn, highlightByReactant=True)
# png = d2d.GetDrawingText()
# open('reaction4.png', 'wb+').write(png)')
d2d = Draw.MolDraw2DCairo(800, 300)
d2d.DrawReaction(rxn, highlightByReactant=True)
png = d2d.GetDrawingText()
open('reaction6.png', 'wb+').write(png)

rxn = AllChem.ReactionFromSmarts('([C:13]-[N;H0;D3;+0:14](-[C:15])-[C;H0;D3;+0:1](=[O;D1;H0:2])-[c:3]1:[c:4]:[c:5](-[C:6](=[O;D1;H0:7])-[#8:8]-[C:9]):[c:10]:[c:11]:[c:12]:1)>>(O-[C;H0;D3;+0:1](=[O;D1;H0:2])-[c:3]1:[c:4]:[c:5](-[C:6](=[O;D1;H0:7])-[#8:8]-[C:9]):[c:10]:[c:11]:[c:12]:1).([C:13]-[NH;D2;+0:14]-[C:15])')
# d2d = Draw.MolDraw2DCairo(800, 300)
# d2d.DrawReaction(rxn, highlightByReactant=True)
# png = d2d.GetDrawingText()
# open('reaction4.png', 'wb+').write(png)')
d2d = Draw.MolDraw2DCairo(800, 300)
d2d.DrawReaction(rxn, highlightByReactant=True)
png = d2d.GetDrawingText()
open('reaction7.png', 'wb+').write(png)

# from rdkit import Chem
# mol = Chem.MolFromSmiles('I[c:16]1[c:14]([OH:15])[c:12]([CH3:13])[cH:11][c:10]([CH2:9][c:8]2[cH:7][cH:6][c:5]([O:4][CH2:3][O:2][CH3:1])[c:20]([CH:21]([CH3:22])[CH3:23])[cH:19]2)[c:17]1[CH3:18]')
# for atom in mol.GetAtoms():
#     print(atom.GetAtomMapNum())
#
#     atom.SetAtomMapNum(0)
#
#     print(atom.GetAtomMapNum())
#
# smi = Chem.MolToSmiles(mol)
# print(smi)
#
# print('Done')








