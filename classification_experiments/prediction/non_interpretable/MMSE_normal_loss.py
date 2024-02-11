import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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


def normalize(train_split, test_split):
    feat_train = train_split[:, :-2]
    lab_train_mmse = train_split[:, -2:-1].astype('int').ravel()
    lab_train_mmse_binned = train_split[:, -1:].astype('int').ravel()

    feat_test = test_split[:, :-2]
    lab_test_mmse = test_split[:, -2:-1].astype('int').ravel()
    lab_test_mmse_binned = test_split[:, -1:].astype('int').ravel()

    # X = StandardScaler().fit_transform(matrix_feat)

    X_train, X_test, y_train, y_test, y_train_binned, y_test_binned = feat_train, feat_test, lab_train_mmse, lab_test_mmse, lab_train_mmse_binned, lab_test_mmse_binned

    X_train = X_train.astype('float')
    X_test = X_test.astype('float')
    normalized_train_X = (X_train - X_train.mean(0)) / (X_train.std(0) + 0.01)
    normalized_test_X = (X_test - X_train.mean(0)) / (X_train.std(0) + 0.01)

    return normalized_train_X, normalized_test_X, y_train, y_test, y_train_binned, y_test_binned


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


def map_values_to_ranges(value_list):
    mapped_values = []
    for value in value_list:
        if 1 <= value <= 3:
            mapped_values.append(0)
        elif 4 <= value <= 6:
            mapped_values.append(1)
        elif 7 <= value <= 9:
            mapped_values.append(2)
        elif 10 <= value <= 12:
            mapped_values.append(3)
        elif 13 <= value <= 15:
            mapped_values.append(4)
        elif 16 <= value <= 18:
            mapped_values.append(5)
        elif 19 <= value <= 21:
            mapped_values.append(6)
        elif 22 <= value <= 24:
            mapped_values.append(7)
        elif 25 <= value <= 27:
            mapped_values.append(8)
        elif 28 <= value <= 30:
            mapped_values.append(9)
        else:
            mapped_values.append(None)  # Handle values outside the specified ranges
    return mapped_values


def rmse_function(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))


# class MMSE_ModelBasic(nn.Module):
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


class MMSE_ModelBasic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MMSE_ModelBasic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 70)
        self.fc3 = nn.Linear(70, 30)
        self.fc4 = nn.Linear(30, 1)  # Output is a single value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


#class MMSEPredictionModel2(nn.Module):  # first predicing bins then real values
#    def __init__(self, input_size, hidden_size, num_bins):
#        super(MMSEPredictionModel2, self).__init__()
#        self.fc1 = nn.Linear(input_size, hidden_size)
#        self.fc2_mmse = nn.Linear(num_bins, 1)  # Output layer for predicting MMSE score directly
#        self.fc2_bins = nn.Linear(hidden_size, num_bins)  # Output layer for predicting binned MMSE interval
#        # 10 = num of bins
#
#    def forward(self, x):
#        x = self.fc1(x)
#        bins_logits = self.fc2_bins(x)
#        bins_probabilities = F.softmax(bins_logits, dim=1)
#        mmse_score = self.fc2_mmse(bins_probabilities)
#        return mmse_score, bins_logits, bins_probabilities
#
#
#class CustomLoss(nn.Module):
#    def __init__(self):
#        super(CustomLoss, self).__init__()
#
#    def forward(self, y_true_mmse, y_true_bins, mmse_pred, bins_logits):
#        # Calculate mean squared error for MMSE score prediction
#        mse_loss = F.mse_loss(mmse_pred, y_true_mmse.float().unsqueeze(1))
#        print(mse_loss)
#        # Calculate cross-entropy loss for binned MMSE interval prediction
#        ce_loss = F.cross_entropy(bins_logits, y_true_bins.long(), reduction='sum')
#
#        loss = (1 / torch.sum(y_true_bins)) * (mse_loss - ce_loss)
#
#        return loss


seed = 19
torch.manual_seed(seed)

feats_names = ['xvector', 'XLM-Roberta-Large-Vit-L-14', 'whisper', 'trillsson',
               'stsb-xlm-r-multilingual', 'distiluse-base-multilingual-cased-v2',
               'distiluse-base-multilingual-cased-v1', 'all-mpnet-base-v2',
               'all-MiniLM-L12-v2', 'all-MiniLM-L6-v2', 'all-distilroberta-v1']

labels_df = pd.read_csv('/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_labels/groundtruth.csv')
lang_id = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/lang_id_train/lang_ids.csv'

for feat_name in feats_names:
    print(f"Experiments with {feat_name}")
    feat_pth_pd = f'/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/embeddings/{feat_name}/'
    out_path = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_regression/non_interpretable/'
    out_path_scores = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/non_interpretable/regression/'
    print(f"The output directory exists--> {os.path.isdir(out_path)}")
    # test_only = 0

    path_files = [os.path.join(feat_pth_pd, elem) for elem in sorted(os.listdir(feat_pth_pd))]
    names = ['taukdial-' + os.path.basename(elem).rsplit('-', -1)[1] for elem in path_files]
    ids = [os.path.basename(elem).rsplit('.npy')[0] + '.wav' for elem in path_files]
    if labels_df['tkdname'].tolist() == ids:
        mmse = labels_df['mmse'].tolist()
        mmse_binned = map_values_to_ranges(mmse)
        labels = labels_df['dx'].tolist()
        labels = [1 if elem == 'NC' else 0 for elem in labels]
        # labels = [1 if elem =='NC' else 0 for elem in labels]
        print('DONE')
    else:
        print('error in the labels order')
    # labels_pd = [0]*len(path_files_pd)
    df_pd = pd.DataFrame(list(zip(names, path_files, mmse, mmse_binned, labels)),
                         columns=['names', 'path_feat', 'mmse', 'mmse_binned', 'labels'])
    #
    arrayOfSpeaker_cn = sorted(list(set(df_pd.groupby('labels').get_group(1)['names'].tolist())))
    ##
    arrayOfSpeaker_pd = sorted(list(set(df_pd.groupby('labels').get_group(0)['names'].tolist())))

    print(arrayOfSpeaker_pd)
    print(arrayOfSpeaker_cn)

    cn_sps = get_n_folds(arrayOfSpeaker_cn)
    pd_sps = get_n_folds(arrayOfSpeaker_pd)
    #
    data = []
    for cn_sp, pd_sp in zip(sorted(cn_sps, key=len), sorted(pd_sps, key=len, reverse=True)):
        data.append(cn_sp + pd_sp)
    n_folds = sorted(data, key=len, reverse=True)

    ## PER FILE
    folds = []
    folds_names = []

    for i in n_folds:
        names = []
        data_fold = np.array(())  # %
        data_i = df_pd[df_pd["names"].isin(i)]
        # % extract features from files
        for index, row in data_i.iterrows():
            label_row = row['mmse']
            label_row_mmse_binned = row['mmse_binned']
            feat = np.load(row['path_feat'])
            path = row['path_feat']
            # print(label_row, row['path_feat'])
            feat = np.append(feat, label_row)  # attach label to the end of array [1, feat dim + 1]
            feat = np.append(feat, label_row_mmse_binned)
            data_fold = np.vstack((data_fold, feat)) if data_fold.size else feat
            names.append(os.path.basename(path).split('.npy')[0])
        folds.append(data_fold)
        folds_names.append(names)

    #
    data_train_1 = np.concatenate(folds[:9])
    data_test_1 = np.concatenate(folds[-1:])
    data_train_2 = np.concatenate(folds[1:])
    data_test_2 = np.concatenate(folds[:1])
    data_train_3 = np.concatenate(folds[2:] + folds[:1])
    data_test_3 = np.concatenate(folds[1:2])
    data_train_4 = np.concatenate(folds[3:] + folds[:2])
    data_test_4 = np.concatenate(folds[2:3])
    data_train_5 = np.concatenate(folds[4:] + folds[:3])
    data_test_5 = np.concatenate(folds[3:4])
    data_train_6 = np.concatenate(folds[5:] + folds[:4])
    data_test_6 = np.concatenate(folds[4:5])
    data_train_7 = np.concatenate(folds[6:] + folds[:5])
    data_test_7 = np.concatenate(folds[5:6])
    data_train_8 = np.concatenate(folds[7:] + folds[:6])
    data_test_8 = np.concatenate(folds[6:7])
    data_train_9 = np.concatenate(folds[8:] + folds[:7])
    data_test_9 = np.concatenate(folds[7:8])
    data_train_10 = np.concatenate(folds[9:] + folds[:8])
    data_test_10 = np.concatenate(folds[8:9])

    data_test_1_names = np.concatenate(folds_names[-1:])
    data_test_2_names = np.concatenate(folds_names[:1])
    data_test_3_names = np.concatenate(folds_names[1:2])
    data_test_4_names = np.concatenate(folds_names[2:3])
    data_test_5_names = np.concatenate(folds_names[3:4])
    data_test_6_names = np.concatenate(folds_names[4:5])
    data_test_7_names = np.concatenate(folds_names[5:6])
    data_test_8_names = np.concatenate(folds_names[6:7])
    data_test_9_names = np.concatenate(folds_names[7:8])
    data_test_10_names = np.concatenate(folds_names[8:9])
    #

    learning_rate = 0.01
    num_epochs = 35
    batch_size = 32
    input_size = data_train_1.shape[1] - 2
    hidden_size = 40
    num_bins = 10
    criterion = nn.MSELoss()
    # criterion = CustomLoss()

    truth = []
    predictions = []
    results = {}
    results_2 = {}

    for n_fold in range(1, 11):
        print(n_fold)
        model = MMSE_ModelBasic(input_size, hidden_size)
        # model = MMSEPredictionModel2(input_size, hidden_size, num_bins)
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # DATA
        Xtrain, Xtest, mmse_labels_train, mmse_labels_test, \
        bins_labels_train, bins_labels_test = normalize(eval(f"data_train_{n_fold}"), eval(f"data_test_{n_fold}"))

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
    avg_mmse = round(sum / len(results.items()), 2)
    print(f'Average R2: {avg_mmse}')

    sum = 0.0
    for key, value in results_2.items():
        #   print(f'Fold {key}: {value} %')
        sum += value
    avg_R2 = round(sum / len(results_2.items()), 2)
    print(f'Average R2: {avg_R2}')

    dict = {'rmse': avg_mmse, 'r2': avg_R2}
    df2 = pd.DataFrame(dict)
    file_out = os.path.join(out_path, feat_name + "_" + ".csv")

    #######################################################################################################

    all_names = list(data_test_1_names) + list(data_test_2_names) + list(data_test_3_names) \
                + list(data_test_4_names) + list(data_test_5_names) + list(data_test_6_names) \
                + list(data_test_7_names) + list(data_test_8_names) + list(data_test_9_names) \
                + list(data_test_10_names)

    dict = {'names': all_names, 'truth': truth, 'predictions': predictions}
    df2 = pd.DataFrame(dict)
    file_out2 = os.path.join(out_path_scores, feat_name + '.csv')
    df2.to_csv(file_out2)
