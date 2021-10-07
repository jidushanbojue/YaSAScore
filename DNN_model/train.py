import os
import pandas as pd
from scipy import sparse
import numpy as np
from rdkit import Chem
from rdkit.DataStructs import cDataStructs
from rdkit.Chem import AllChem
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import Sequence
import keras
from keras import regularizers
import joblib


def smiles_to_ecfp(product, size=2048):
    """Converts a single SMILES into an ECFP4
    :parameter:
        product (str): The SMILES string corresponding to the product/molecule of choice.
        size (int): Size (dimension) of the ECFP4 vector to be calculated.
    Return:
        ecfp4 (arr): An n dimensional ECFP4 vector of the molecule
    """
    try:
        mol = Chem.MolFromSmiles(product)
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=size)
        arr = np.zeros((0,), dtype=np.int8)
        cDataStructs.ConvertToNumpyArray(ecfp, arr)
    except:
        arr = ''
    return arr


class FPSequnce(Sequence):
    def __init__(self, input_matrix, label_matrix, batch_size):
        self.input_matrix = input_matrix
        self.label_matrix = label_matrix
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.label_matrix.shape[0] / float(self.batch_size)))

    def __getitem__(self, item):
        X_input = self.input_matrix[item*self.batch_size: (item+1)*self.batch_size]
        Y_input = self.label_matrix[item*self.batch_size: (item+1)*self.batch_size]

        return (X_input.todense(), Y_input)
        # return (X_input, Y_input)

def generate_dataset(src_file):
    """
    :param src_file: containing the smiles and targets
    :return: data_input, data_label
    """
    df = pd.read_csv(src_file)
    # df.columns = ['id', 'smiles', 'targets']
    input_list = [x for x in df['smiles'].apply(smiles_to_ecfp, size=2048).values]
    input_csr = sparse.csr_matrix(input_list)
    input_label = keras.utils.to_categorical(np.array(df['targets']).reshape(len(df), 1), num_classes=2)
    print('Done')
    return input_csr, input_label


def DNN_ECFP_training(train_file, val_file, result_path, project_name):

    # generate_dataset(train_file)

    train_inputs, train_labels_one_hot = generate_dataset(train_file)
    val_inputs, val_labels_one_hot = generate_dataset(val_file)
    # test_inputs, test_labels_one_hot = generate_dataset(test_file)

    batch_size = 1024
    nb_epochs = 500


    print('Building model ...')
    model = Sequential()
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    # model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    # model.add(BatchNormalization())
    model.add(Dense(2))
    model.add(Activation('softmax'))

    early_stopping = EarlyStopping(monitor='val_loss', patience=20)

    csv_logger = CSVLogger(os.path.join(result_path, project_name+'.log'), append=True)

    # csv_logger = CSVLogger('24w_training_last_log_ecfp_3_3_0.0001_3_layers.log', append=True)


    model_check = os.path.join(result_path, project_name+'.hdf5')
    checkpoint = ModelCheckpoint(model_check, monitor='loss', save_best_only=True)

    # checkpoint = ModelCheckpoint('24w_last_min_ecfp_weights_3_3_0.0001_3layers.hdf5', monitor='loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=5,
                                  verbose=0,
                                  mode='auto',
                                  min_delta=0.000001,
                                  cooldown=0,
                                  min_lr=0)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    history = model.fit(x=FPSequnce(train_inputs, train_labels_one_hot, batch_size=batch_size),
                        epochs=nb_epochs,
                        steps_per_epoch=train_inputs.shape[0] // batch_size,
                        callbacks=[early_stopping, csv_logger, checkpoint, reduce_lr],
                        validation_data=FPSequnce(val_inputs, val_labels_one_hot,
                                                  batch_size=batch_size),
                        verbose=1)

    print('done!!!')


if __name__ == '__main__':


    #### all data #####
    import argparse
    parser = argparse.ArgumentParser(description='Genearte DNN ECFP model')
    parser.add_argument('--train_file', type=str, help='Specify the train file')
    parser.add_argument('--val_file', type=str, help='Specify the val file')
    # parser.add_argument('--test_file', type=str, help='Specify the test file')
    parser.add_argument('--save_path', type=str, help='Specify the result directory')
    parser.add_argument('--project_name', type=str, help='Specify the project name')
    parser.add_argument('--gpu_index', type=str, help='Specify the GPU number to user')

    # args = parser.parse_args([
    #     '--train_file', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/24w_train_df_seed0.csv',
    #     '--val_file', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/24w_val_df_seed0.csv',
    #     # '--test_file', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/24w_test_df_seed0.csv',
    #     '--save_path', '/data/baiqing/PycharmProjects/dnn_model/split_by_3',
    #     '--project_name', 'split_3'
    #     '--gpu_index', '0'
    # ])

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    ##### ECFP  ####

    train_file = args.train_file
    val_file = args.val_file
    # test_file = args.test_file

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    project_name = args.project_name

    ### test set from balanced set
    # test_file = os.path.join(base_dir, '24w_test_df_seed0.csv')

    ### test set from imbalanced set
    # test_file = os.path.join(base_dir, '24w_cmpnn_remain_all_test.csv')

    # result_picture = os.path.join(script_dir, '19w_min_ecfp_3_3_0.0001_3layers_all_test.png')


    DNN_ECFP_training(train_file, val_file, save_path, project_name)
    ######################
