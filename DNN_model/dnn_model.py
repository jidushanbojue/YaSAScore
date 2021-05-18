import os
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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

        # return (X_input.todense(), Y_input)
        return (X_input, Y_input)

def DNN_predict(model, data_set, result_picture):
    test_inputs_top35w = data_set[0]
    test_inputs_35w_last = data_set[1]
    test_labels_top35w = data_set[2]
    test_labels_35w_last = data_set[3]

    policy = keras.models.load_model(model)
    # loss, accuracy = policy.evaluate(x=FPSequnce(test_inputs, test_labels, batch_size=1), verbose=1)
    pred_arr_top35w = policy.predict(x=FPSequnce(test_inputs_top35w, test_labels_top35w, batch_size=1), verbose=1)
    pred_arr_35w_last = policy.predict(x=FPSequnce(test_inputs_35w_last, test_labels_35w_last, batch_size=1), verbose=1)

    pred_arr = np.concatenate([pred_arr_top35w, pred_arr_35w_last])
    test_labels = np.concatenate([test_labels_top35w, test_labels_35w_last])

    pred = np.argmax(pred_arr, axis=1)

    c = confusion_matrix(test_labels[:, 1], pred)
    matt = matthews_corrcoef(test_labels[:, 1], pred)
    # acc_score = accuracy_score(test_labels[: 1], pred)
    print('predict the test set is {}'.format(pred))
    # print('Test score is {}'.format(accuracy))
    print('confusion matrix is {}'.format(c))
    print('matthews_corrcoef is {}'.format(matt))
    # print('acc_score is {}'.format(accuracy))

    fpr, tpr, threshold = roc_curve(test_labels[:, 1], pred_arr[:, 1])
    roc_auc = auc(fpr, tpr)

    font1 = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 23}

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 25}

    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='red', lw=5)
    plt.xlabel('False positive rate', fontdict=font1)
    plt.ylabel('True positive rate', fontdict=font1)
    plt.title('DNN classifier (AUC={})'.format(roc_auc), fontdict=font2)
    # plt.legend('R^2 of test: {}'.format(test_score))
    # plt.text(0.5, 0.6, 'Test score is {}'.format(accuracy))
    plt.text(0.5, 0.5, 'matthews corrcoef is {}'.format(matt))
    # plt.text(0.5, 0.4, 'accuracy score is {}'.format(accuracy))
    plt.text(0.5, 0.3, 'confusion corrcoef is {}'.format(c))

    # plt.savefig('ECFP_from_all_data_RF_roc_AUC1_n_estimator1000_oobTrue_mean.png')
    plt.savefig(result_picture)
    plt.show()

    print('Done!!!')
    print('Done!!!')
    # return loss, accuracy, pred


def generate_dataset(src_file):
    """
    :param src_file: containing the smiles and targets
    :return: data_input, data_label
    """
    df = pd.read_csv(src_file)
    input_list = [x for x in df['smiles'].apply(smiles_to_ecfp, size=2048).values]
    input_csr = sparse.csr_matrix(input_list)
    input_label = keras.utils.to_categorical(np.array(df['p_np']).reshape(len(df), 1), num_classes=2)
    print('Done')
    return input_csr, input_label


def DNN_ECFP_training(train_file, val_file, test_file, result_picture):

    # generate_dataset(train_file)

    # train_inputs, train_labels_one_hot = generate_dataset(train_file)
    # val_inputs, val_labels_one_hot = generate_dataset(val_file)
    test_inputs, test_labels_one_hot = generate_dataset(test_file)


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

    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    csv_logger = CSVLogger('24w_training_last_log_ecfp_3_3_0.0001_3_layers.log', append=True)
    # checkpoint_loc = os.path.join(os.path.split(result_picture)[0], 'checkpoints')
    # os.mkdir(checkpoint_loc)
    checkpoint = ModelCheckpoint('24w_last_min_ecfp_weights_3_3_0.0001_3layers.hdf5', monitor='loss', save_best_only=True)
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


    # history = model.fit(x=FPSequnce(train_inputs, train_labels_one_hot, batch_size=batch_size),
    #                     epochs=nb_epochs,
    #                     steps_per_epoch=train_inputs.shape[0] // batch_size,
    #                     callbacks=[early_stopping, csv_logger, checkpoint, reduce_lr],
    #                     validation_data=FPSequnce(val_inputs, val_labels_one_hot,
    #                                               batch_size=batch_size),
    #                     verbose=1)

    pred = DNN_predict('24w_last_min_ecfp_weights_3_3_0.0001_3layers.hdf5', (test_inputs, test_labels_one_hot), result_picture)

    # score = model.predict(x=batch_generator(test_inputs, np.array(test_labels).reshape(len(test_labels), 1), batch_size=1), steps=test_inputs.shape[0], verbose=1)
    # score = model.predict(x=FPSequnce(test_inputs, np.array(test_labels).reshape(len(test_labels), 1), batch_size=1), verbose=1)
    print('done!!!')


if __name__ == '__main__':


    #### all data #####

    base_dir = 'projects/data'
    script_dir = os.path.dirname(os.path.abspath(__file__))


    ##### ECFP  ####
    train_file = os.path.join(base_dir, '24w_train_df_seed0.csv')
    val_file = os.path.join(base_dir, '24w_val_df_seed0.csv')

    ### test set from balanced set
    # test_file = os.path.join(base_dir, '24w_test_df_seed0.csv')

    ### test set from imbalanced set
    test_file = os.path.join(base_dir, '24w_cmpnn_remain_all_test.csv')

    result_picture = os.path.join(script_dir, '19w_min_ecfp_3_3_0.0001_3layers_all_test.png')


    DNN_ECFP_training(train_file, val_file, test_file, result_picture)
    ######################
