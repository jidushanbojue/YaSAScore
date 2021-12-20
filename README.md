# YaSAScore
Source code for our  paper "Prediction of compound synthesis accessibility based on reaction knowledge graph"

The code was built based on CMPNN (https://github.com/SY575/CMPNN), SYBA(https://github.com/lich-uct/syba). Thanks a lot for their code sharing!

Prediction results of compound synthesis accessibility (SA) based on the a refined chemical network constructed on the USPTO and Pistachio reaction datasets. 

###**The performance of different models on the balanced test set**

Partition Criterion | MODEL |  AUC  |  ACC  | MCC
:---: | :---: |:-----:|:-----:| :---:
SRP:2 | DNN-PCD | 0.576 | 0.511 | 0.044
SRP:2 | RF-PCD | 0.632 | 0.592 | 0.183
SRP:2 | CMPNN | 0.784 | 0.711 | 0.432
SRP:2 | DNN-ECFP | 0.72  | 0.667 | 0.333
SRP:2 | SYBA | 0.491 | 0.505 | 0.02
SRP:2 | SYBA-2 | 0.739 | 0.668 | 0.343
SRP:2 | SAScore | 0.535 | 0.501 | -0.03
SRP:2 | SCScore | 0.613 | 0.55  | 0.128
SRP:3 | DNN-PCD | 0.615 | 0.584 | 0.168
SRP:3 | RF-PCD | 0.627 | 0.588 | 0.177
SRP:3 | CMPNN | 0.791 | 0.715 | 0.434
SRP:3 | DNN-ECFP | 0.751 | 0.687 | 0.373
SRP:3 | SYBA | 0.465 | 0.496 | -0.012
SRP:3 | SYBA-2 | 0.76  | 0.69  | 0.382
SRP:3 | SAScore | 0.513 |  0.5  | -0.011
SRP:3 | SCScore | 0.621 | 0.543 | 0.116
SRP:4 | DNN-PCD |  0.6  | 0.565 | 0.132
SRP:4 | RF-PCD | 0.627 | 0.583 | 0.168
SRP:4 | CMPNN | 0.814 | 0.733 | 0.466
SRP:4 | DNN-ECFP | 0.802 | 0.732 | 0.465
SRP:4 | SYBA | 0.448 | 0.491 | -0.061
SRP:4 | SYBA-2 |  0.8  | 0.727 | 0.453
SRP:4 | SAScore | 0.45  | 0.512 | -0.021
SRP:4 | SCScore | 0.591 | 0.517 | 0.082


![network](file:///home/cadd/Desktop/ranting/24w_property_kdeplot_ps.png "network")

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
    python get_reaction_steps.py -gf ../../data/chemical_reaction_network/uspto_graph_reverse.graph -df ../../data/chemical_reaction_network/degree.csv -rf ../../data/chemical_reaction_network/reaction_to_structure_USPTO_test.csv -o ../../data/chemical_reaction_network/shortest_path
    
note: some big file such as reaction_all_structure_uspto, reaction_to_structure_USPTO.csv, uspto_graph_with_relationship.graphml are available at [google drive](https://drive.google.com/drive/folders/18zyTaHIgmmG0C2dnm8BDISYOPNW1Jhi0)

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
    python predict.py --data_path ../data/cmpnn_data/24w_test_df_seed0.csv --checkpoint_dir ckpt_for_3_split

### SYBA-2 training
    cd sascore_scscore_syba_syba2_model/scipt
    conda env create -f syba_environment.yaml ### Create env
    conda activate syba_env ### activate env
    cd script
    python syba-2_training.py --HS_train ../../data/syba_data/24w_train_HS.csv
                              --ES_train ../../data/syba_data/24w_train_ES.csv 
                              --count_file ../../data/syba_data/syba_ES_cluster_HS_train_val.csv 
                              --score_file ../../data/syba_data/syba_ES_cluster_HS_score_train_val.csv  #### Get count_file and score_file
    cp ../../data/syba_data/syba_ES_cluster_HS_train_val.csv ../syba/resources ### Then put the score result "syba_ES_cluster_HS_score_train_val.csv" to syba/resources

    
    Note: AS when training SYBA, separate (ES and HS) file are needed, that are 24w_train_HS.csv and 24w_train_ES.csv. 
    The two files are recomined from 24w_train_df_seed0.csv and 24w_val_df_seed0.csv. The purpose of doing this is to provide the comparative result of different models
    

### syba, syba-2, sascore, scscore predict
    cd sascore_scscore_syba_syba2_model/scipt
    
    ### Get the scoring value of 24w_test_file
    python diff_score_result.py --in_file ../../data/syba_data/24w_test_df_seed0.csv
                                --out_file ../../data/syba_data/24w_test_df_seed0_syba_and_mysyba.csv
    
    #### Get the scoring value of all_remain_test_file (63w)
    python diff_score_result.py --in_file ../../data/syba_data/24w_cmpnn_remain_all_test.csv
                                --out_file ../../data/syba_data/24w_test_df_seed0_syba_and_mysyba_all_test.csv
    
   
    
### DNN training and predict
    Note: If split by shortest reaction steps 3, train file is dnn_data/24w_train_df_seed0.csv, validate file is dnn_data/24w_val_df_seed0.csv, test file is dnn_data/24w_test_df_seed0.csv
          When adding the remained test items, the test file is dnn_data/24w_cmpnn_remain_all_test.csv
    
          If split by shortest reaction steps 2, train file is dnn_data/60w_train_df_seed0.csv, validate file is dnn_data/60w_val_df_seed0.csv, test file is dnn_data/60w_test_df_seed0.csv
          When adding the remained test items, the test file is dnn_data/60w_cmpnn_remain_all_test_2_split.csv

          If split by shortest reaction steps 4, train file is dnn_data/8w_train_df_seed0.csv, validate file is dnn_data/8w_val_df_seed0.csv, test file is dnn_data/8w_test_df_seed0.csv
          When adding the remained test items, the test file is dnn_data/8w_cmpnn_remain_all_test_4_split.csv
    
    cd DNN_model
    python train.py --train_file ../../data/dnn_data/24w_train_df_seed0.csv
                    --val_file ../../data/dnn_data/24w_val_df_seed0.csv
                    --save_path ../../data/dnn_data/split_by_3
                    --project_name split_3
                    --gpu_index 0
    
    python predict.py --model_path ../../data/dnn_data/split_by_3/split_3.hdf5
                      --test_file  ../../data/dnn_data/24w_test_df_seed0.csv
                      --save_path  ../../data/dnn_data/split_by_3
                      --project_name split_3

### View ES and HS distribution 
    cd picture
    python get_ES_HS_file.py --train_file ../data/cmpnn_data/24w_train_df_seed0.csv
                             --val_file   ../data/cmpnn_data/24w_val_df_seed0.csv
                             --ES_out     ../data/cmpnn_data/24w_ES.csv
                             --HS_out     ../data/cmpnn_data/24w_HS.csv
    
    python generate_physicochemical_property.py --ES_file ../data/cmpnn_data/24w_ES.csv
                                                --HS_file ../data/cmpnn_data/24w_HS.csv
                                                --out     ../data/cmpnn_data/24w_ES_HS_property.csv
    
    python plot_physicochemical_property.py --in_file ../data/cmpnn_data/24w_ES_HS_property.csv
                                            --threshold 3
                                            --out 24w_property_kdeplot.png

    python chemical_space_all_training_data.py --train_file ../data/cmpnn_data/24w_train_df_seed0.csv
                                               --val_file   ../data/cmpnn_data/24w_val_df_seed0.csv
                                               --test_file  ../data/cmpnn_data/24w_test_df_seed0.csv
                                               --pca_result  ../data/cmpnn_data/24w_pca_result.csv
                                               --threshold   3
                                               --out         24w_pca_picture.png
    
    
          



   