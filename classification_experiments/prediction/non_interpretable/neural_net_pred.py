import pandas as pd
import os
import nltk
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np
import pandas as pd
import random
import numpy as np
import random
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
# coding: utf-8
import sys

import pandas as pd
import numpy as np
from statistics import mode
from sklearn.utils import shuffle
import os
import sys
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
random_state = 20
random_seed = 20
#np.random.seed(20)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
test_only = 0


# !/usr/bin/env python
# coding: utf-8

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


# Define your model
class SimpleNet(nn.Module):
    def __init__(self, input_size):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # Fully connected layer
        self.layer_norm = nn.LayerNorm(1)  # Layer normalization
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):
        x = self.fc(x)
        x = self.layer_norm(x)
        x = self.sigmoid(x)
        return x


class ComplexNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ComplexNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, 1)  # Second fully connected layer
        self.layer_norm1 = nn.LayerNorm(hidden_size)  # Layer normalization for first hidden layer
        self.layer_norm2 = nn.LayerNorm(1)  # Layer normalization for second hidden layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = self.sigmoid(x)
        return x


feats_names = ['xvector', 'XLM-Roberta-Large-Vit-L-14', 'whisper', 'trillsson',
               'stsb-xlm-r-multilingual', 'distiluse-base-multilingual-cased-v2',
               'distiluse-base-multilingual-cased-v1', 'all-mpnet-base-v2',
               'all-MiniLM-L12-v2', 'all-MiniLM-L6-v2', 'all-distilroberta-v1']

labels_df = pd.read_csv('/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_labels/groundtruth.csv')
lang_id = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/lang_id_train/lang_ids.csv'

for feat_name in feats_names:
    print(f"Experiments with {feat_name}")
    feat_pth_pd = f'/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/embeddings/{feat_name}/'
    # out_path = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_classification/non_interpretable/'
    #  print(f"The output directory exists--> {os.path.isdir(out_path)}")
    # test_only = 0

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
    random.Random(random_seed).shuffle(arrayOfSpeaker_cn)
    ##
    arrayOfSpeaker_pd = sorted(list(set(df_pd.groupby('labels').get_group(0)['names'].tolist())))
    random.Random(random_seed).shuffle(arrayOfSpeaker_pd)

    print(arrayOfSpeaker_pd)
    print(arrayOfSpeaker_cn)

    cn_sps = get_n_folds(arrayOfSpeaker_cn)
    pd_sps = get_n_folds(arrayOfSpeaker_pd)
    #
    data = []
    for cn_sp, pd_sp in zip(sorted(cn_sps, key=len), sorted(pd_sps, key=len, reverse=True)):
        data.append(cn_sp + pd_sp)
    n_folds = sorted(data, key=len, reverse=True)
    # print(n_folds)

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

    # Set random seed for reproducibility
    num_epochs = 50
    learning_rate = 0.0001
    batch_size = 32  # Change this to match the dimensionality of your embeddings
    hidden_size = 50
    # Define loss function
    criterion = nn.BCELoss()

    # Perform cross-validation
    fold_accuracies = []
    for i in range(1, 11):
        print(i)
        normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}"),
                                                                           eval(f"data_test_{i}"))
        #  y_val = y_val.tolist()
        input_size = normalized_train_X[0].shape[0]
        print(input_size)
        # Convert inputs and labels to tensors
        X_train_tensor = torch.FloatTensor(normalized_train_X)
        print(X_train_tensor.shape)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        print(X_train_tensor.shape)
        X_val_tensor = torch.FloatTensor(normalized_test_X)
        print(X_val_tensor.shape)
        y_val_tensor = torch.FloatTensor(y_test).view(-1, 1)

        # Initialize model
        # model = ComplexNet(input_size, hidden_size)
        model = SimpleNet(input_size)

        # Define optimizer for each fold
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for j in range(0, len(X_train_tensor), batch_size):  # Use a different variable name for the inner loop
                inputs = X_train_tensor[j:j + batch_size]
                labels = y_train_tensor[j:j + batch_size]

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            #
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}")

        # Validation
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_tensor)
            predicted_labels = (outputs > 0.5).float()
            accuracy = accuracy_score(y_val_tensor, predicted_labels)
            print(f"Validation Accuracy: {accuracy}")
            fold_accuracies.append(accuracy)

    # Calculate and print average cross-validation accuracy
    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f"Average Cross-Validation Accuracy: {avg_accuracy}")


