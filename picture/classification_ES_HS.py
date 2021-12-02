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


class FPSequence(Sequence):
    def __init__(self, input_arr, label_arr, batch_size):
        self.input_arr = input_arr
        self.label_arr = label_arr
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.label_arr.shape[0] / float(self.batch_size)))

    def __getitem__(self, item):
        X_input = self.input_arr[item*self.batch_size: (item+1)*self.batch_size]
        Y_input = self.label_arr[item*self.batch_size: (item+1)*self.batch_size]
        return (X_input, Y_input)

def cal_descriptor(df, result_file):
    molwt_list = []
    tpsa_list = []
    nrotb_list = []
    hbd_list = []
    hba_list = []
    logp_list = []
    new_df = pd.DataFrame()
    for idx, line in df.iterrows():
        print(idx)
        mol = Chem.MolFromSmiles(line['smiles'])
        molwt_list.append(Descriptors.ExactMolWt(mol))
        tpsa_list.append(Chem.rdMolDescriptors.CalcTPSA(mol))
        nrotb_list.append(Descriptors.NumRotatableBonds(mol))
        hbd_list.append(Descriptors.NumHDonors(mol))
        hba_list.append(Descriptors.NumHAcceptors(mol))
        logp_list.append(Descriptors.MolLogP(mol))
    new_df['MolWt'] = molwt_list
    new_df['TPSA'] = tpsa_list
    new_df['nRotB'] = nrotb_list
    new_df['HBD'] = hbd_list
    new_df['HBA'] = hba_list
    new_df['LogP'] = logp_list
    new_df.to_csv(result_file)
    return np.array(new_df), np.array(df['targets'])


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

    # font1 = {'family': 'Times New Roman',
    #         'weight': 'normal',
    #         'size': 23}
    #
    # font2 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 25}

    # plt.figure(figsize=(10, 10))
    # plt.plot(fpr, tpr, color='red', lw=5)
    # plt.xlabel('False positive rate', fontdict=font1)
    # plt.ylabel('True positive rate', fontdict=font1)
    # plt.title('DNN classifier (AUC={})'.format(roc_auc), fontdict=font2)
    # # plt.legend('R^2 of test: {}'.format(test_score))
    # # plt.text(0.5, 0.6, 'Test score is {}'.format(accuracy))
    # plt.text(0.5, 0.5, 'matthews corrcoef is {}'.format(matt))
    # # plt.text(0.5, 0.4, 'accuracy score is {}'.format(accuracy))
    # plt.text(0.5, 0.3, 'confusion corrcoef is {}'.format(c))
    #
    # # plt.savefig('ECFP_from_all_data_RF_roc_AUC1_n_estimator1000_oobTrue_mean.png')
    # plt.savefig(result_picture)
    # plt.show()
    #
    # print('Done!!!')
    # print('Done!!!')
    # # return loss, accuracy, pred

def DNN_descriptor_training(train_file, val_file, test_file, result_picture):

    # generate_dataset(train_file)


    df_train = pd.read_csv(train_file)
    # train_desc, train_value = cal_descriptor(df_train, 'train_six_desc.csv')
    train_desc_df = pd.read_csv('train_six_desc.csv')
    del train_desc_df['Unnamed: 0']


    df_val = pd.read_csv(val_file)
    # val_desc, val_value = cal_descriptor(df_val, 'val_six_desc.csv')
    val_desc_df = pd.read_csv('val_six_desc.csv')
    del val_desc_df['Unnamed: 0']

    df_test = pd.read_csv(test_file)
    # test_desc, test_value = cal_descriptor(df_test, 'test_six_desc.csv')
    test_desc_df = pd.read_csv('test_six_desc.csv')
    del test_desc_df['Unnamed: 0']

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

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    csv_logger = CSVLogger('/home/baiqing/24w_training_last_log_six_descriptor_3_3_0.0001_3_layers_1.log', append=True)

    checkpoint = ModelCheckpoint('/home/baiqing/24w_last_min_six_descriptor_weights_3_3_0.0001_3layers_1.hdf5', monitor='loss', save_best_only=True)
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


    train_arr = np.array(train_desc_df)
    train_value = keras.utils.to_categorical(np.array(df_train['targets']).reshape(len(df_train), 1), num_classes=2)

    val_arr = np.array(val_desc_df)
    val_value = keras.utils.to_categorical(np.array(df_val['targets']).reshape(len(df_val), 1), num_classes=2)

    test_arr = np.array(test_desc_df)
    test_value = keras.utils.to_categorical(np.array(df_test['targets']).reshape(len(df_test), 1), num_classes=2)



    # history = model.fit(x=train_arr,
    #                     y=train_value,
    #                     epochs=nb_epochs,
    #                     # steps_per_epoch=train_desc_df.shape[0]//batch_size,
    #                     callbacks=[early_stopping, csv_logger, checkpoint, reduce_lr],
    #                     validation_data=(val_arr, val_value),
    #                     verbose=1
    #                     )


    pred = DNN_predict('/home/baiqing/24w_last_min_six_descriptor_weights_3_3_0.0001_3layers_1.hdf5', (test_arr, test_value), result_picture)


    print('done!!!')

def RF_six_training(train_file, train_desc_file, test_file, test_desc_file):
    """
    :param src_file: contain molecular name (ES_id or HS_id) and smile string
    :return:
    """
    train_df = pd.read_csv(train_file, )
    # del train_df['Unnamed: 0']
    train_desc_df = pd.read_csv(train_desc_file)
    del train_desc_df['Unnamed: 0']

    test_df = pd.read_csv(test_file)
    # del test_df['Unnamed: 0']
    test_desc_df = pd.read_csv(test_desc_file)
    del test_desc_df['Unnamed: 0']
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, oob_score=True, n_jobs=100)
    clf.fit(train_desc_df, train_df['targets'])
    joblib.dump(clf, '24w_RF_six_descriptor.m')

    y_prob = clf.predict_proba(test_desc_df)
    pred = clf.predict(test_desc_df)
    accuracy = clf.score(test_desc_df, test_df['targets'])
    matt = matthews_corrcoef(test_df['targets'], pred)
    fpr, tpr, threshold = roc_curve(test_df['targets'], y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    # stats_list.append({'pred': pred,
    #                     'accuracy': accuracy,
    #                     'matthews_corrcoef': matt,
    #                     'roc_auc': roc_auc,
    #                     'n_estimator': n_estimator})
    print('Done')




if __name__ == '__main__':

    base_dir = '/data/baiqing/PycharmProjects/YaSAScore/data/dnn_data/'
    script_dir = os.path.dirname(os.path.abspath(__file__))


    ###### ECFP  ####
    train_file = os.path.join(base_dir, '24w_train_df_seed0.csv')
    val_file = os.path.join(base_dir, '24w_val_df_seed0.csv')
    test_file = os.path.join(base_dir, '24w_test_df_seed0.csv')
    # test_file = os.path.join(base_dir, '24w_cmpnn_remain_all_test.csv')
    result_picture = os.path.join(script_dir, '19w_min_ecfp_3_3_0.0001_3layers_all_test.png')
    # DNN_descriptor_training(train_file, val_file, test_file, result_picture)

    train_desc_file = 'train_six_desc.csv'
    val_desc_file = 'val_six_desc.csv'
    test_desc_file = 'test_six_desc.csv'

    RF_six_training(train_file, train_desc_file, test_file, test_desc_file)



