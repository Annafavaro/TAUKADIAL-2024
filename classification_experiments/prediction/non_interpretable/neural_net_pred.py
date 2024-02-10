import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score

torch.manual_seed(19)


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
    train_set = train_split
    test_set = test_split

    feat_train = train_set[:, :-1]
    lab_train = train_set[:, -1:]
    lab_train = lab_train.astype('int')

    feat_test = test_set[:, :-1]
    lab_test = test_set[:, -1:]
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


class Classifier(nn.Module):
   def __init__(self,  input_dim):
       super().__init__()
       self.hidden = nn.Linear(input_dim, 180)
       self.relu = nn.ReLU()
       self.output = nn.Linear(180, 1)
       self.sigmoid = nn.Sigmoid()
   def forward(self, x):
       x = self.relu(self.hidden(x))
       x = self.sigmoid(self.output(x))
       return x


#class Classifier(nn.Module):
#    def __init__(self, input_dim):
#        super().__init__()
#        # Increase depth and width
#        self.hidden1 = nn.Linear(input_dim, 256)
#        self.bn1 = nn.BatchNorm1d(256)  # Batch normalization
#        self.hidden2 = nn.Linear(256, 128)
#        self.bn2 = nn.BatchNorm1d(128)
#        self.hidden3 = nn.Linear(128, 64)
#        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
#        self.output = nn.Linear(64, 1)
#        # Using LeakyReLU as an alternative to ReLU
#        self.leaky_relu = nn.LeakyReLU(0.01)
#
#    def forward(self, x):
#        x = self.leaky_relu(self.bn1(self.hidden1(x)))
#        x = self.leaky_relu(self.bn2(self.hidden2(x)))
#        x = self.dropout(x)
#        x = self.leaky_relu(self.hidden3(x))
#        x = torch.sigmoid(self.output(x))
#        return x
#

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #  print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


feats_names = ['xvector', 'XLM-Roberta-Large-Vit-L-14', 'whisper', 'trillsson',
               'stsb-xlm-r-multilingual', 'distiluse-base-multilingual-cased-v2',
               'distiluse-base-multilingual-cased-v1', 'all-mpnet-base-v2',
               'all-MiniLM-L12-v2', 'all-MiniLM-L6-v2', 'all-distilroberta-v1']

labels_df = pd.read_csv('/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_labels/groundtruth.csv')
lang_id = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/lang_id_train/lang_ids.csv'

for feat_name in feats_names:
    print(f"Experiments with {feat_name}")
    feat_pth_pd = f'/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/embeddings/{feat_name}/'

    path_files = [os.path.join(feat_pth_pd, elem) for elem in sorted(os.listdir(feat_pth_pd))]
    names = ['taukdial-' + os.path.basename(elem).rsplit('-', -1)[1] for elem in path_files]
    ids = [os.path.basename(elem).rsplit('.npy')[0] + '.wav' for elem in path_files]
    if labels_df['tkdname'].tolist() == ids:
        print('equal')
        labels = labels_df['dx'].tolist()
        labels = [1 if elem == 'NC' else 0 for elem in labels]
        print('DONE')
    else:
        print('error in the labels order')
    # labels_pd = [0]*len(path_files_pd)
    df_pd = pd.DataFrame(list(zip(names, path_files, labels)), columns=['names', 'path_feat', 'labels'])
    #
    arrayOfSpeaker_cn = sorted(list(set(df_pd.groupby('labels').get_group(1)['names'].tolist())))
    # random.Random(random_seed).shuffle(arrayOfSpeaker_cn)
    ##
    arrayOfSpeaker_pd = sorted(list(set(df_pd.groupby('labels').get_group(0)['names'].tolist())))
    #    random.Random(random_seed).shuffle(arrayOfSpeaker_pd)

    print(arrayOfSpeaker_pd)
    print(arrayOfSpeaker_cn)

    cn_sps = get_n_folds(arrayOfSpeaker_cn)
    pd_sps = get_n_folds(arrayOfSpeaker_pd)
    #
    data = []
    for cn_sp, pd_sp in zip(sorted(cn_sps, key=len), sorted(pd_sps, key=len, reverse=True)):
        data.append(cn_sp + pd_sp)
    n_folds = sorted(data, key=len, reverse=True)

    #
    ## PER FILE
    folds = []
    for i in n_folds:
        names = []
        data_fold = np.array(())  # %
        data_i = df_pd[df_pd["names"].isin(i)]
        # % extract features from files
        for index, row in data_i.iterrows():
            label_row = row['labels']
            feat = np.load(row['path_feat'])
            path = row['path_feat']
            # print(label_row, row['path_feat'])
            feat = np.append(feat, label_row)  # attach label to the end of array [1, feat dim + 1]
            data_fold = np.vstack((data_fold, feat)) if data_fold.size else feat
            names.append(path)
        folds.append(data_fold)

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

    n_epochs = 30
    batch_size = 32
    input_dim = data_train_1.shape[1] - 1  # Subtract 1 for the label column
    print(input_dim)
    hidden_dim = 256  # Hidden dimension of the fully connected layer
    output_dim = 1  # Output dimension for binary classification (1 for binary)
    learning_rate = 0.01
    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss

    results = {}
    for n_fold in range(1, 11):
        print(n_fold)
        model = Classifier(input_dim)
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # DATA
        Xtrain, Xtest, y_train, y_test = normalize(eval(f"data_train_{n_fold}"), eval(f"data_test_{n_fold}"))
       # print(len(Xtrain), len(Xtest))
        batches_per_epoch = len(Xtrain) // batch_size

        for epoch in range(n_epochs):
            model.train()
            for i in range(batches_per_epoch):
                optimizer.zero_grad()
                start = i * batch_size
                # take a batch
                Xbatch = Xtrain[start:start + batch_size]
                ybatch = y_train[start:start + batch_size]
                Xbatch = torch.tensor(Xbatch, dtype=torch.float32)
                ybatch = torch.tensor(ybatch, dtype=torch.float32)

                # forward pass
                y_pred = model(Xbatch)
                loss = criterion(y_pred.flatten(), ybatch)
                # backward pass
                acc = (y_pred.round() == ybatch).float().mean()
                loss.backward()
                # update weights
                optimizer.step()
        #  print(f"epoch {epoch} step {i} loss {loss} accuracy {acc}")

        # evaluate trained model with test set
        correct, total = 0, 0
        with torch.no_grad():
            Xtest = torch.tensor(Xtest, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
            y_pred = model(Xtest)
        accuracy = (y_pred.round() == y_test).float().mean()

        # Print accuracy
        # print(f'Accuracy for fold {n_fold} is {accuracy}')
        # print('--------------------------------')
        results[n_fold] = accuracy

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR 10 FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum / len(results.items())} %')
