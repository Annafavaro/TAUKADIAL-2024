feats_names = ['DINO', 'XLM-Roberta-Large-Vit-L-14', 'lealla-base', 'multilingual-e5-large', 'text2vec-base-multilingual',
               'distiluse-base-multilingual-cased', 'distiluse-base-multilingual-cased-v1',
               'bert-base-multilingual-cased', 'LaBSE', 'wav2vec_128', 'wav2vec_53', 'whisper', 'trillsson', 'xvector']


english_sps = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_speaker_division_helin/en.json'
lang_id = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/lang_id_train/lang_ids.csv'
path_labels = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_labels/groundtruth.csv'
feat_pths = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/embeddings/'

out_path_scores ='/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language/english/non_interpretable/regression/'
out_path = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_per_language/english/regression/non_interpretable/'

import json
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import torch
import sklearn
from sklearn.metrics import r2_score
seed = 19
torch.manual_seed(seed)


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #  print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def get_n_folds(arrayOfSpeaker):
    data = list(arrayOfSpeaker)  # list(range(len(arrayOfSpeaker)))
    num_of_folds = 10
    n_folds = []
    for i in range(num_of_folds):
        n_folds.append(data[int(i * len(data) / num_of_folds):int((i + 1) * len(data) / num_of_folds)])
    return n_folds

def normalize(train_split, test_split): ## when prediction
    train_set = train_split
    test_set = test_split

    feat_train = train_set[:, :-2]
    lab_train = train_set[:, -1:]
    lab_train = lab_train.astype('int')

    feat_test = test_set[:, :-2]
    lab_test = test_set[:, -1:] #-1 is where MMSE are
    lab_test = lab_test.astype('int')

    # X = StandardScaler().fit_transform(matrix_feat)

    X_train, X_test, y_train, y_test = feat_train, feat_test, lab_train, lab_test
    y_test = y_test.ravel()
    y_train = y_train.ravel()
    X_train = X_train.astype('float')
    X_test = X_test.astype('float')
    normalized_test_X = (X_test - X_train.mean(0)) / (X_train.std(0) + 0.01)
    normalized_train_X = (X_train - X_train.mean(0)) / (X_train.std(0) + 0.01)

    return normalized_train_X, normalized_test_X, y_train, y_test


def add_labels(df, path_labels):
    path_labels_df = pd.read_csv(path_labels)
    label = path_labels_df['dx'].tolist()
    speak = path_labels_df['tkdname'].tolist()  # id
    spk2lab_ = {sp: lab for sp, lab in zip(speak, label)}
    speak2__ = df['ID'].tolist()
    etichettex = []
    for nome in speak2__:
        if nome in spk2lab_.keys():
            lav = spk2lab_[nome]
            etichettex.append(([nome, lav]))
        else:
            etichettex.append(([nome, 'Unknown']))
    label_new_ = []
    for e in etichettex:
        label_new_.append(e[1])
    df['labels'] = label_new_

    return df


def rmse_function(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))


class MMSE_ModelBasic(nn.Module):
    def __init__(self, input_size):
        super(MMSE_ModelBasic, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.fc(x)
        return x.squeeze(1)

#class MMSE_ModelBasic(nn.Module):
#    def __init__(self, input_size, hidden_size):
#        super(MMSE_ModelBasic, self).__init__()
#        self.fc1 = nn.Linear(input_size, hidden_size)
#        self.fc2 = nn.Linear(hidden_size, 1)  # Output is a single value
#
#    def forward(self, x):
#        x = torch.relu(self.fc1(x))
#        x = self.fc2(x)
#        return x
#

for feat_name in feats_names:
    print(f"Experiments with {feat_name}")

    n_folds_names = []
    n_folds_data = []
    all_folds_info = []

    read_dict = json.load(open(english_sps))
    for key, values in read_dict.items():
        fold_info_general = []
        fold = list((read_dict[key]).keys())
        n_folds_names.append(list([os.path.basename(sp) for sp in fold]))
        fold_info = read_dict[key]  # get data for
        for sp in fold_info:
            fold_info_general.append(
                [os.path.join(feat_pths, feat_name, sp.split('.wav')[0] + '.npy'), (fold_info[sp])['label'],
                 (fold_info[sp])['mmse']])
        all_folds_info.append(fold_info_general)

    print(n_folds_names[0])
    folds = []
    for fold in all_folds_info:
        data_fold = np.array(())  # %
        for speaker in fold:
            label_row = speaker[-2]
            mmse = speaker[-1]
            feat = np.load(speaker[0])
            # print(label_row, row['path_feat'])
            feat = np.append(feat, label_row)
            feat = np.append(feat, mmse)
            data_fold = np.vstack((data_fold, feat)) if data_fold.size else feat
        folds.append(data_fold)

    print(folds[0])

    # For fold 1
    data_train_1 = np.concatenate(folds[:8])
    data_val_1 = np.concatenate(folds[8:9])
    data_test_1 = np.concatenate(folds[9:])

    # For fold 2
    data_train_2 = np.concatenate((folds[1:-1]))
    data_val_2 = np.concatenate(folds[-1:])
    data_test_2 = np.concatenate(folds[:1])

    # For fold 3
    data_train_3 = np.concatenate(folds[2:])
    data_val_3 = np.concatenate(folds[:1])
    data_test_3 = np.concatenate(folds[1:2])

    # For fold 4
    data_train_4 = np.concatenate((folds[3:] + folds[:1]))
    data_val_4 = np.concatenate(folds[1:2])
    data_test_4 = np.concatenate(folds[2:3])

    # For fold 5
    data_train_5 = np.concatenate((folds[4:] + folds[:2]))
    data_val_5 = np.concatenate(folds[2:3])
    data_test_5 = np.concatenate(folds[3:4])

    # For fold 6
    data_train_6 = np.concatenate((folds[5:] + folds[:3]))
    data_val_6 = np.concatenate(folds[3:4])
    data_test_6 = np.concatenate(folds[4:5])

    # For fold 7
    data_train_7 = np.concatenate((folds[6:] + folds[:4]))
    data_val_7 = np.concatenate(folds[4:5])
    data_test_7 = np.concatenate(folds[5:6])

    # For fold 8
    data_train_8 = np.concatenate((folds[7:] + folds[:5]))
    data_val_8 = np.concatenate(folds[5:6])
    data_test_8 = np.concatenate(folds[6:7])

    # For fold 9
    data_train_9 = np.concatenate((folds[8:] + folds[:6]))
    data_val_9 = np.concatenate(folds[6:7])
    data_test_9 = np.concatenate(folds[7:8])

    # For fold 10
    data_train_10 = np.concatenate((folds[9:] + folds[:7]))
    data_val_10 = np.concatenate(folds[7:8])
    data_test_10 = np.concatenate(folds[8:9])

    data_test_1_names = np.concatenate(n_folds_names[-1:])
    data_test_2_names = np.concatenate(n_folds_names[:1])
    data_test_3_names = np.concatenate(n_folds_names[1:2])
    data_test_4_names = np.concatenate(n_folds_names[2:3])
    data_test_5_names = np.concatenate(n_folds_names[3:4])
    data_test_6_names = np.concatenate(n_folds_names[4:5])
    data_test_7_names = np.concatenate(n_folds_names[5:6])
    data_test_8_names = np.concatenate(n_folds_names[6:7])
    data_test_9_names = np.concatenate(n_folds_names[7:8])
    data_test_10_names = np.concatenate(n_folds_names[8:9])

    #data_train_1 = np.concatenate(folds[:9])
    #data_test_1 = np.concatenate(folds[-1:])
    #data_train_2 = np.concatenate(folds[1:])
    #data_test_2 = np.concatenate(folds[:1])
    #data_train_3 = np.concatenate(folds[2:] + folds[:1])
    #data_test_3 = np.concatenate(folds[1:2])
    #data_train_4 = np.concatenate(folds[3:] + folds[:2])
    #data_test_4 = np.concatenate(folds[2:3])
    #data_train_5 = np.concatenate(folds[4:] + folds[:3])
    #data_test_5 = np.concatenate(folds[3:4])
    #data_train_6 = np.concatenate(folds[5:] + folds[:4])
    #data_test_6 = np.concatenate(folds[4:5])
    #data_train_7 = np.concatenate(folds[6:] + folds[:5])
    #data_test_7 = np.concatenate(folds[5:6])
    #data_train_8 = np.concatenate(folds[7:] + folds[:6])
    #data_test_8 = np.concatenate(folds[6:7])
    #data_train_9 = np.concatenate(folds[8:] + folds[:7])
    #data_test_9 = np.concatenate(folds[7:8])
    #data_train_10 = np.concatenate(folds[9:] + folds[:8])
    #data_test_10 = np.concatenate(folds[8:9])
#
    #data_test_1_names = np.concatenate(n_folds_names[-1:])
    #data_test_2_names = np.concatenate(n_folds_names[:1])
    #data_test_3_names = np.concatenate(n_folds_names[1:2])
    #data_test_4_names = np.concatenate(n_folds_names[2:3])
    #data_test_5_names = np.concatenate(n_folds_names[3:4])
    #data_test_6_names = np.concatenate(n_folds_names[4:5])
    #data_test_7_names = np.concatenate(n_folds_names[5:6])
    #data_test_8_names = np.concatenate(n_folds_names[6:7])
    #data_test_9_names = np.concatenate(n_folds_names[7:8])
    #data_test_10_names = np.concatenate(n_folds_names[8:9])

    learning_rate = 0.01
    num_epochs = 20
    batch_size = 32
    input_size = data_train_1.shape[1] - 2
    hidden_size = 20
    criterion = nn.MSELoss()
    # criterion = CustomLoss()

    truth = []
    predictions = []
    results = {}
    results_2 = {}

    for n_fold in range(1, 11):
        print(n_fold)
        model = MMSE_ModelBasic(input_size)
        #model = MMSE_ModelBasic(input_size, hidden_size)
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # DATA
        Xtrain, Xtest, mmse_labels_train, mmse_labels_test = normalize(eval(f"data_train_{n_fold}"), eval(f"data_test_{n_fold}"))

        print(len(Xtrain), len(Xtest))
        batches_per_epoch = len(Xtrain) // batch_size

        for epoch in range(num_epochs):

            model.train()
            total_loss = 0.0
            total_mmse_rmse = 0.0
            # total_accuracy = 0.0

            for i in range(batches_per_epoch):
                optimizer.zero_grad()
                start = i * batch_size
                # take a batch
                Xbatch = Xtrain[start:start + batch_size]
                y_train_batch_mmse = mmse_labels_train[start:start + batch_size]
                # y_train_batch_mmse_binned = bins_labels_train[start:start+batch_size]
                Xbatch = torch.tensor(Xbatch, dtype=torch.float32)
                y_train_batch_mmse = torch.tensor(y_train_batch_mmse, dtype=torch.float32)
                outputs = model(Xbatch)
                # Compute loss
                loss = criterion(outputs.squeeze(), y_train_batch_mmse)
                # Backward pass
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_mmse_rmse += rmse_function(outputs, y_train_batch_mmse).item()

        avg_train_loss = total_loss / len(Xtrain)
        print(f'training loss:{avg_train_loss}')
        avg_train_mmse_rmse = total_mmse_rmse / len(Xtrain)

        correct, total = 0, 0
        model.eval()
        total_val_loss = 0.0
        total_val_mmse_rmse = 0.0
        total_val_accuracy = 0.0
        with torch.no_grad():
            Xtest = torch.tensor(Xtest, dtype=torch.float32)
            y_test_mmse = torch.tensor(mmse_labels_test, dtype=torch.float32)
            # y_test_batch_mmse_binned = torch.tensor(bins_labels_test, dtype=torch.float32)
            y_pred = model(Xtest)  # .detach().numpy()
        rmse_val = rmse_function(y_pred.squeeze(), y_test_mmse)
        r2_val = r2_score(y_test_mmse.numpy(), y_pred.squeeze().numpy())
        truth = truth + list(y_test_mmse.numpy())
        predictions = predictions + list(y_pred.squeeze().numpy())

        results[n_fold] = rmse_val
        results_2[n_fold] = r2_val

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR 10 FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        #   print(f'Fold {key}: {value} %')
        sum += value
    avg_mmse = sum / len(results.items())
    print(f'Average MMSE: {avg_mmse}')

    sum = 0.0
    for key, value in results_2.items():
        sum += value
    avg_R2 = sum / len(results_2.items())
    print(f'Average R2: {avg_R2}')

    dict = {'rmse': avg_mmse, 'r2': avg_R2}
    df = pd.DataFrame(dict, index=[0])
    file_out = os.path.join(out_path, feat_name + "_" + ".csv")
    df.to_csv(file_out)

    #######################################################################################################

    all_names = list(data_test_1_names) + list(data_test_2_names) + list(data_test_3_names) \
                + list(data_test_4_names) + list(data_test_5_names) + list(data_test_6_names) \
                + list(data_test_7_names) + list(data_test_8_names) + list(data_test_9_names) \
                + list(data_test_10_names)

    dict = {'names': all_names, 'truth': truth, 'predictions': predictions}
    df2 = pd.DataFrame(dict)
    file_out2 = os.path.join(out_path_scores, feat_name + '.csv')
    df2.to_csv(file_out2)
