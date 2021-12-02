import os

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import Sequence
import keras
from keras import regularizers

import numpy as np

def DNN_predict(model, data_set, result_picture):
    # test_inputs_top35w = data_set[0]
    # test_inputs_35w_last = data_set[1]
    # test_labels_top35w = data_set[2]
    # test_labels_35w_last = data_set[3]

    test_arr = data_set[0]
    test_value = data_set[1]
    policy = keras.models.load_model(model)
    pred_arr = policy.predict(x=test_arr, verbose=1)

    pred = np.argmax(pred_arr, axis=1)

    c = confusion_matrix(test_value[:, 1], pred)
    acc = accuracy_score(test_value[:, 1], pred)
    print('Accuracy is ', acc)
    matt = matthews_corrcoef(test_value[:, 1], pred)
    # acc_score = accuracy_score(test_value[: 1], pred)
    print('predict the test set is {}'.format(pred))
    # print('Test score is {}'.format(accuracy))
    print('confusion matrix is {}'.format(c))
    print('matthews_corrcoef is {}'.format(matt))
    # print('acc_score is {}'.format(accuracy))

    fpr, tpr, threshold = roc_curve(test_value[:, 1], pred_arr[:, 1])
    roc_auc = auc(fpr, tpr)
    print('AUC of ROC is ', roc_auc)

def DNN_descriptor_training(train_desc_file, val_desc_file, result_path, project_name):


    # train_desc, train_value = cal_descriptor(df_train, 'train_six_desc.csv')
    train_desc_df = pd.read_csv(train_desc_file)
    del train_desc_df['Unnamed: 0']

    # val_desc, val_value = cal_descriptor(df_val, 'val_six_desc.csv')
    val_desc_df = pd.read_csv(val_desc_file)
    del val_desc_df['Unnamed: 0']

    # test_desc, test_value = cal_descriptor(df_test, 'test_six_desc.csv')
    # test_desc_df = pd.read_csv(test_desc_file)
    # del test_desc_df['Unnamed: 0']

    batch_size = 1024
    nb_epochs = 500

    print('Building model ...')
    model = Sequential()
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Dense(2))
    model.add(Activation('softmax'))

    early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    csv_logger = CSVLogger(os.path.join(result_path, project_name+'.log'), append=True)

    model_check = os.path.join(result_path, project_name+'.hdf5')
    checkpoint = ModelCheckpoint(model_check, monitor='loss', save_best_only=True)
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


    train_arr = np.array(train_desc_df[['MolWt', 'TPSA', 'nRotB', 'HBD', 'HBA', 'LogP']])
    train_value = keras.utils.to_categorical(np.array(train_desc_df['targets']).reshape(len(train_desc_df), 1), num_classes=2)

    val_arr = np.array(val_desc_df[['MolWt', 'TPSA', 'nRotB', 'HBD', 'HBA', 'LogP']])
    val_value = keras.utils.to_categorical(np.array(val_desc_df['targets']).reshape(len(val_desc_df), 1), num_classes=2)

    # test_arr = np.array(test_desc_df)
    # test_value = keras.utils.to_categorical(np.array(test_desc_df['targets']).reshape(len(test_desc_df), 1), num_classes=2)



    history = model.fit(x=train_arr,
                        y=train_value,
                        epochs=nb_epochs,
                        steps_per_epoch=train_arr.shape[0]//batch_size,
                        callbacks=[early_stopping, csv_logger, checkpoint, reduce_lr],
                        validation_data=(val_arr, val_value),
                        verbose=1
                        )


    # pred = DNN_predict('/home/baiqing/24w_last_min_six_descriptor_weights_3_3_0.0001_3layers_1.hdf5', (test_arr, test_value), result_picture)


    print('done!!!')


if __name__ == '__main__':
    #### all data #####
    import argparse
    parser = argparse.ArgumentParser(description='Genearte DNN six descriptor model')
    parser.add_argument('--train_file', type=str, help='Specify the train file')
    parser.add_argument('--val_file', type=str, help='Specify the val file')
    parser.add_argument('--save_path', type=str, help='Specify the result directory')
    parser.add_argument('--project_name', type=str, help='Specify the project name')
    parser.add_argument('--gpu_index', type=str, help='Specify the GPU number to user')

    # args = parser.parse_args([
    #     '--train_file', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/24w_train_six_desc.csv',
    #     '--val_file', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/24w_val_six_desc.csv',
    #     '--save_path', '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/split_by_3_physicochem',
    #     '--project_name', 'split_3_physicochem',
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



    DNN_descriptor_training(train_file, val_file, save_path, project_name)
