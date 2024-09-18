feats_names = ['XLM-Roberta-Large-Vit-L-14', 'lealla-base',
               'multilingual-e5-large', 'whisper',
               'text2vec-base-multilingual', 'xlm-roberta-base',
               'distiluse-base-multilingual-cased',
               'distiluse-base-multilingual-cased-v1', 'DINO',
               'bert-base-multilingual-cased', 'LaBSE', 'wav2vec_128',
               'wav2vec_53', 'trillsson', 'xvector']

english_sps = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_speaker_division_helin/zh.json'
lang_id = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/lang_id_train/lang_ids.csv'
path_labels = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_labels/groundtruth.csv'
feat_pths = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/embeddings/'
out_path_scores ='/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language/chinese/non_interpretable/regression/'
out_path = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_per_language/chinese/regression/non_interpretable/'

import json
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import torch
import sklearn
from sklearn.metrics import r2_score

seed = 44
torch.manual_seed(seed)

class MMSE_ModelBasic(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super(MMSE_ModelBasic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Function to reset neural network weights
def reset_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# Function to compute RMSE
def rmse_function(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))


def normalize(train_split, val_split, test_split):  ## when prediction
    train_set = train_split
    val_set = val_split
    test_set = test_split

    feat_train = train_set[:, :-2]
    lab_train = train_set[:, -1:]
    lab_train = lab_train.astype('int')

    feat_val = val_set[:, :-2]
    lab_val = val_set[:, -1:]
    lab_val = lab_val.astype('int')

    feat_test = test_set[:, :-2]
    lab_test = test_set[:, -1:]  # -1 is where MMSE are
    lab_test = lab_test.astype('int')

    # X = StandardScaler().fit_transform(matrix_feat)

    X_train, X_val, X_test = feat_train, feat_val, feat_test
    y_train, y_val, y_test = lab_train, lab_val, lab_test
    y_test = y_test.ravel()
    y_train = y_train.ravel()
    y_val = y_val.ravel()

    X_train = X_train.astype('float')
    X_val = X_val.astype('float')
    X_test = X_test.astype('float')

    # Normalize the features
    mean_train = X_train.mean(0)
    std_train = X_train.std(0) + 0.01
    normalized_train_X = (X_train - mean_train) / std_train
    normalized_val_X = (X_val - mean_train) / std_train
    normalized_test_X = (X_test - mean_train) / std_train

    return normalized_train_X, normalized_val_X, normalized_test_X, y_train, y_val, y_test


def rmse_function(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))


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

    learning_rate = 0.01
    num_epochs = 300
    batch_size = 64
    input_size = data_train_1.shape[1] - 2
    hidden_size = 60
    criterion = nn.MSELoss()

    truth = []
    predictions = []
    results = {}
    results_2 = {}
    rmse_vals = []

    for n_fold in range(1, 11):
        print(n_fold)
        # model = MMSE_ModelBasic(input_size)
        model = MMSE_ModelBasic(input_size, hidden_size)
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # DATA
        Xtrain, Xval, Xtest, mmse_labels_train, mmse_labels_val, mmse_labels_test = normalize(
            eval(f"data_train_{n_fold}"), eval(f"data_val_{n_fold}"), eval(f"data_test_{n_fold}"))

        # print(len(Xtrain), len(Xtest))
        batches_per_epoch = len(Xtrain) // batch_size
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            print(f'epoch-->{epoch}')
            model.train()
            total_loss = 0.0
            total_mmse_rmse = 0.0
            patience = 5
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
                # print(loss)
                # Backward pass
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_mmse_rmse += rmse_function(outputs, y_train_batch_mmse).item()

            avg_train_loss = total_loss / len(Xtrain)
            #print(f'training loss:{avg_train_loss}')
            avg_train_mmse_rmse = total_mmse_rmse / len(Xtrain)

            model.eval()
            with torch.no_grad():
                Xval_tensor = torch.tensor(Xval, dtype=torch.float32)
                y_val_tensor = torch.tensor(mmse_labels_val, dtype=torch.float32)
                y_val_pred = model(Xval_tensor)
                val_loss = criterion(y_val_pred.flatten(), y_val_tensor)
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                num_epochs_no_improve = 0
            else:
                num_epochs_no_improve += 1
                if num_epochs_no_improve >= patience:
                  #  print(f"Early stopping at epoch {epoch}")
                    break

        # Testing
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32)
            y_test_mmse = torch.tensor(mmse_labels_test, dtype=torch.float32)
            y_pred = model(Xtest_tensor)
        rmse_val = rmse_function(y_pred.squeeze(), y_test_mmse)
        rmse_vals.append(rmse_val)
        r2_val = r2_score(y_test_mmse.numpy(), y_pred.squeeze().numpy())

        # Store ground truth and predictions
        truth.extend(y_test_mmse.numpy())
        predictions.extend(y_pred.squeeze().numpy())

    rmse_tot = np.mean(rmse_vals)
    print(f'final_rmse_tot {rmse_tot}')

    dict = {'rmse': rmse_tot}
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