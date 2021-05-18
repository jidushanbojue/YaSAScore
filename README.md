# YaSAScore
Source code for our  paper "Prediction of compound synthesis accessibility based on reaction knowledge graph"

The code was built based on CMPNN (https://github.com/SY575/CMPNN), SYBA(https://github.com/lich-uct/syba). Thanks a lot for their code sharing!

Prediction results of compound synthesis accessibility (SA) based on the a refined chemical network constructed on the USPTO and Pistachio reaction datasets. 

##Quick start

###Template extract
    python Template_Extraction_and_Validation.py -d uspto_pistachio_split_folder -o uspto_pistachio_result -f template -r 1


### Generate chemical_reaction_network and get minimum reaction steps
    mkdir uspto_reaction_split
    cd uspto_reaction_split

    split -l 1000 data/chemical_reaction_network/all_reaction_uspto.csv -d -a 4

    python generate_reaction_structure_relationship.py -d uspto_reaction_split -o uspto_reaction_split_result -s data/chemical_reaction_network/all_structure_uspto.csv

    "combining the spliting files in uspto_reaction_split_result

    python generate_network_multiprocess.py
    
    python get_reaction_steps.py -gf uspto_graph_with_relationship_reverse.graphml -df node_degree_with_relationship_uspto.csv -rf reaction_to_structure_USPTO.csv -o result_folder

###CMPNN training
    `python train.py -data_path data/cmpnn_data/24w_cmpnn.csv --dataset_type classification --num_folder 1 --gpu 0 --epochs 30`
    `python predict.py --data_path data/cmpnn_data/24w_cmpnn_df_seed0.csv --checkpoint_dir ckpt_epochs_30

###SYBA training
    `



   