# YaSAScore
Source code for our  paper "Prediction of compound synthesis accessibility based on reaction knowledge graph"

The code was built based on CMPNN (https://github.com/SY575/CMPNN), SYBA(https://github.com/lich-uct/syba). Thanks a lot for their code sharing!

Prediction results of compound synthesis accessibility (SA) based on the a refined chemical network constructed on the USPTO and Pistachio reaction datasets. 

MODEL | ROC-AUC | ACC | MCC
---  | :---: | :---: | :---:
CMPNN| 0.791 | 0.715 | 0.434
DNN-ECFP|0.749|0.685 | 0.371
SYBA | 0.465 | 0.497 | -0.012
SYBA-2|0.76  | 0.69  | 0.382
SAScore|0.513| 0.498 | -0.011
SCScore|0.621|0.582  | 0.167

## Quick start

### Template extract
    cd template_extract
    conda env create -f template_extract.yaml  ### create env
    conda activate casp_env_tf2   ### change env
    mkdir uspto_pistachio_split_folder
    mkdir uspto_pistachio_result
    cd uspto_pistachio_split_folder
    split -l 100 ../uspto_and_pistachio_top100.csv -d -a 3 uspto_pistachio_
    cd ..
    python Template_Extraction_and_Validation.py -d uspto_pistachio_split_folder -o uspto_pistachio_result -f template -r 1


### Generate chemical_reaction_network and get minimum reaction steps
    cd chemical_reaction_network_graph
    mkdir uspto_reaction_split
    cd uspto_reaction_split
    split -l 1000 ../../data/chemical_reaction_network/all_reaction_uspto.csv -d -a 4
    cd ..
    python generate_reaction_structure_relationship.py -d uspto_reaction_split -o uspto_reaction_split_result -s ../data/chemical_reaction_network/all_structure_uspto.csv
    cd uspto_reaction_split_result
    nawk 'FNR==1 && NR!=1{next;}{print}' * > ../../data/chemical_reaction_network/reaction_to_structure_USPTO.csv #### combine the split relation file to reaction_to_structure_USPTO.csv
    cd ../../
    python generate_network_multiprocess.py -i ../data/chemical_reaction_network/reaction_to_structure_USPTO.csv -o ../data/chemical_reaction_network/uspto_graph_with_relationship.graphml -ro ../data/chemical_reaction_network/uspto_graph_with_relationship_reverse.graphml -d ../data/chemical_reaction_network/uspto_graph_degree.csv -n_cpu 10
    python get_reaction_steps.py -gf ../../yasascore_test/data/chemical_reaction_network/uspto_graph_reverse.graph -df ../../yasascore_test/data/chemical_reaction_network/degree.csv -rf ../../yasascore_test/data/chemical_reaction_network/reaction_to_structure_USPTO_test.csv -o ../../yasascore_test/data/chemical_reaction_network/shortest_path

### CMPNN training and predict
    cd CMPNN-master
    conda env create -f cmpnn.yaml ### Create env
    conda activate cmpnn
#### Traing process by apply cmpnn model (without cross-validatte)    
    python train.py --data_path ../data/cmpnn_data/24w_cmpnn.csv 
                    --dataset_type classification 
                    --num_folds 1 
                    --gpu 0 
                    --seed 0 
                    --epochs 30 
                    --train_csv ../data/cmpnn_data/24w_train_df_seed0.csv 
                    --validate_csv ../data/cmpnn_data/24w_val_df_seed0.csv 
                    --test_csv ../data/cmpnn_data/24w_test_df_seed0.csv
    python predict.py --data_path ../data/cmpnn_data/24w_cmpnn_df_seed0.csv --checkpoint_dir ckpt_epochs_30

### SYBA-2 training
    cd sascore_scscore_syba_syba2_model/scipt
    conda env create -f syba_environment.yaml ### Create env
    conda activate syba_env ### activate env
    cd script
    python syba-2_training.py --HS_train ../../data/syba_data/24w_train_HS.csv
                              --ES_train ../../data/syba_data/24w_train_ES.csv 
                              --count_file ../../data/syba_data/syba_ES_cluster_HS_train_val.csv 
                              --score_file ../../data/syba_data/syba_ES_cluster_HS_score_train_val.csv  #### Get count_file and score_file
    cp ../../data/syba_data/syba_ES_cluster_HS_score_train_val.csv ../syba/resources ### Then put the score result "syba_ES_cluster_HS_score_train_val.csv" to syba/resources

    
    Note: AS when training SYBA, separate (ES and HS) file are needed, that are 24w_train_HS.csv and 24w_train_ES.csv. 
    The two files are recomined from 24w_train_df_seed0.csv and 24w_val_df_seed0.csv. The purpose of doing this is to provide the comparative result of different models
    

### syba, syba-2, sascore, scscore predict
    cd sascore_scscore_syba_syba2_model/scipt
    python diff_score_result.py
    to get the predicted result of syba, syba-2, sascore, scscore models respectively.
    
### DNN training and predict
    cd DNN_model
    python dnn_model.py
    



   