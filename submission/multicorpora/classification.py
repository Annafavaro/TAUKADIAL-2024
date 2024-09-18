import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import json

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def normalize_and_split_no_test(train_split, val_split, test_split):

    train_set = train_split
    val_set = val_split
    test_set = test_split

    feat_train = train_set[:, :-1]
    lab_train = train_set[:, -1:]
    lab_train = lab_train.astype('int')

    control_group = train_set[train_set[:, -1] == 1]
    control_group = control_group[:, :-1]  # remove labels from features CNs
    median = np.median(control_group, axis=0)
    std = np.std(control_group, axis=0)
    #
    feat_val = val_split[:, :-1]
    lab_val = val_split[:, -1:]
    lab_val = lab_val.astype('int')

    feat_test = test_set

    X_train, X_val, X_test, y_train, y_val = feat_train, feat_val, feat_test, lab_train, lab_val
    y_train = y_train.ravel()
    y_val = y_val.ravel()

    X_train = X_train.astype('float')
    X_val = X_val.astype('float')
    X_test = X_test.astype('float')

    normalized_train_X = (X_train - median) / (std + 0.01)
    normalized_val_X = (X_val - median) / (std + 0.01)
    normalized_test_X = (X_test - median) / (std + 0.01)

    return normalized_train_X, normalized_val_X, normalized_test_X, y_train, y_val


def prepare_test_set(base_path_feat, feat_name, lang_id_file):

    base_names = []
    data_fold = np.array(())
    test_set_path_embeddings = os.path.join(base_path_feat, feat_name)
    all_files_names = [elem.split('.npy')[0] for elem in os.listdir(test_set_path_embeddings)]
    all_files_to_keep = sorted(intersection(all_files_names, lang_id_file))
    all_files_path = [os.path.join(test_set_path_embeddings, elem + '.npy') for elem in sorted(all_files_to_keep)]
    # base_names = [os.path.basename(elem).split('.npy')[0] for elem in sorted(all_files_path)]
    for feat in sorted(all_files_path):
        base_names.append(os.path.basename(feat).split('.npy')[0])
        feat_read = np.load(feat)
        data_fold = np.vstack((data_fold, feat_read)) if data_fold.size else feat_read

    return data_fold, base_names


def predict_average_scores(list_of_lists):
    # Calculate the number of lists
    num_lists = len(list_of_lists)
    print(num_lists)

    # Calculate the length of each list (assuming all lists have the same length)
    list_length = len(list_of_lists[0])
    print(list_length)
    # Initialize a list to store the averages
    averages = []

    # Iterate through each index
    for i in range(list_length):
        # Calculate the sum of elements at the current index across all lists
        total = sum(lst[i] for lst in list_of_lists)
        # Calculate the average
        average = total / num_lists
        # Append the average to the averages list
        averages.append(average)

    # Predict based on the averages
    predictions = [1 if avg > 0.5 else 0 for avg in averages]

    return predictions, averages


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


class SingleLayerClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleLayerClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 70)
        self.fc2 = nn.Linear(70, 30)
        self.fc3 = nn.Linear(30, output_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x.squeeze(1)

feats_names = ['vgg' ,'gpt4', 'XLM-Roberta-Large-Vit-L-14',
               'lealla-base', 'multilingual-e5-large',
               'text2vec-base-multilingual', 'xlm-roberta-base',
               'distiluse-base-multilingual-cased',
               'distiluse-base-multilingual-cased-v1',
               'bert-base-multilingual-cased', 'LaBSE', 'wav2vec_128',
               'wav2vec_53', 'whisper', 'trillsson', 'xvector']


english_sps = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_speaker_division_helin/en.json'
chinese_sps = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_speaker_division_helin/zh.json'
lang_id_test = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/testing/language_id/lang_id.csv'
feat_pths_train = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/embeddings/'
feat_pths_test = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/testing/feats/embeddings/'
out_path_scores = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/testing/submission/non_interpretable/multicorpora/'

lang_id_test_r = pd.read_csv(lang_id_test)
lang_id_test_r_keep_en = lang_id_test_r[lang_id_test_r['lang']=='en']['names'].tolist()
lang_id_test_r_keep_zh = lang_id_test_r[lang_id_test_r['lang']=='zh']['names'].tolist()

seed = 120
torch.manual_seed(seed)

for feat_name in feats_names:
    print(feat_name)

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
                [os.path.join(feat_pths_train, feat_name, sp.split('.wav')[0] + '.npy'), (fold_info[sp])['label']])
        all_folds_info.append(fold_info_general)

    #  print(n_folds_names[0])
    folds = []
    for fold in all_folds_info:
        data_fold = np.array(())  # %
        for speaker in fold:
            label_row = speaker[-1]
            feat = np.load(speaker[0])
            feat = np.append(feat, label_row)
            data_fold = np.vstack((data_fold, feat)) if data_fold.size else feat
        folds.append(data_fold)

    data_test_en, speaker_id_en = prepare_test_set(feat_pths_test, feat_name, lang_id_test_r_keep_en)

    data_train_1_en = np.concatenate(folds[:9])
    print(data_train_1_en.shape)
    data_val_1_en = np.concatenate(folds[-1:])
    print(data_val_1_en.shape)
    data_test_1_en = data_test_en
    print(data_test_1_en.shape)
    #
    data_train_2_en = np.concatenate((folds[1:]))
    data_val_2_en = np.concatenate(folds[:1])
    data_test_2_en = data_test_en

    # For fold 3
    data_train_3_en = np.concatenate(folds[2:] + folds[:1])
    data_val_3_en = np.concatenate(folds[1:2])
    data_test_3_en = data_test_en

    # For fold 4
    data_train_4_en = np.concatenate((folds[3:] + folds[:2]))
    data_val_4_en = np.concatenate(folds[2:3])
    data_test_4_en = data_test_en
    # For fold 5
    data_train_5_en = np.concatenate((folds[4:] + folds[:3]))
    data_val_5_en = np.concatenate(folds[3:4])
    data_test_5_en = data_test_en

    # For fold 6
    data_train_6_en = np.concatenate((folds[5:] + folds[:4]))
    data_val_6_en = np.concatenate(folds[4:5])
    data_test_6_en = data_test_en
    # For fold 7
    data_train_7_en = np.concatenate((folds[6:] + folds[:5]))
    data_val_7_en = np.concatenate(folds[5:6])
    data_test_7_en = data_test_en

    # For fold 8
    data_train_8_en = np.concatenate((folds[7:] + folds[:5]))
    data_val_8_en = np.concatenate(folds[5:6])
    data_test_8_en = data_test_en

    # For fold 9
    data_train_9_en = np.concatenate((folds[8:] + folds[:7]))
    data_val_9_en = np.concatenate(folds[7:8])
    data_test_9_en = data_test_en

    # For fold 10
    data_train_10_en = np.concatenate((folds[9:] + folds[:8]))
    data_val_10_en = np.concatenate(folds[8:9])
    data_test_10_en = data_test_en

    ##########################################################################

    n_folds_names_zh = []
    n_folds_data_zh = []
    all_folds_info_zh = []

    read_dict = json.load(open(chinese_sps))
    for key, values in read_dict.items():
        fold_info_general_zh = []
        fold = list((read_dict[key]).keys())
        n_folds_names_zh.append(list([os.path.basename(sp) for sp in fold]))
        fold_info = read_dict[key]  # get data for
        for sp in fold_info:
            fold_info_general_zh.append(
                [os.path.join(feat_pths_train, feat_name, sp.split('.wav')[0] + '.npy'), (fold_info[sp])['label']])
        all_folds_info_zh.append(fold_info_general_zh)

    #  print(n_folds_names[0])
    folds_zh = []
    for fold in all_folds_info_zh:
        data_fold_zh = np.array(())  # %
        for speaker in fold:
            label_row = speaker[-1]
            feat = np.load(speaker[0])
            # print(label_row, row['path_feat'])
            feat = np.append(feat, label_row)
            data_fold_zh = np.vstack((data_fold_zh, feat)) if data_fold_zh.size else feat
        folds_zh.append(data_fold_zh)

    data_test_zh, speaker_id_zh = prepare_test_set(feat_pths_test, feat_name, lang_id_test_r_keep_zh)

    data_train_1_zh = np.concatenate(folds_zh[:9])
    print(data_train_1_zh.shape)
    data_val_1_zh = np.concatenate(folds_zh[-1:])
    print(data_val_1_zh.shape)
    data_test_1_zh = data_test_zh
    print(data_test_1_zh.shape)
    #
    data_train_2_zh = np.concatenate((folds_zh[1:]))
    data_val_2_zh = np.concatenate(folds_zh[:1])
    data_test_2_zh = data_test_zh

    # For fold 3
    data_train_3_zh = np.concatenate(folds_zh[2:] + folds_zh[:1])
    data_val_3_zh = np.concatenate(folds_zh[1:2])
    data_test_3_zh = data_test_zh

    # For fold 4
    data_train_4_zh = np.concatenate((folds_zh[3:] + folds_zh[:2]))
    data_val_4_zh = np.concatenate(folds_zh[2:3])
    data_test_4_zh = data_test_zh

    # For fold 5
    data_train_5_zh = np.concatenate((folds_zh[4:] + folds_zh[:3]))
    data_val_5_zh = np.concatenate(folds_zh[3:4])
    data_test_5_zh = data_test_zh

    # For fold 6
    data_train_6_zh = np.concatenate((folds_zh[5:] + folds_zh[:4]))
    data_val_6_zh = np.concatenate(folds_zh[4:5])
    data_test_6_zh = data_test_zh

    # For fold 7
    data_train_7_zh = np.concatenate((folds_zh[6:] + folds_zh[:5]))
    data_val_7_zh = np.concatenate(folds_zh[5:6])
    data_test_7_zh = data_test_zh

    # For fold 8
    data_train_8_zh = np.concatenate((folds_zh[7:] + folds_zh[:5]))
    data_val_8_zh = np.concatenate(folds_zh[5:6])
    data_test_8_zh = data_test_zh

    # For fold 9
    data_train_9_zh = np.concatenate((folds_zh[8:] + folds_zh[:7]))
    data_val_9_zh = np.concatenate(folds_zh[7:8])
    data_test_9_zh = data_test_zh

    # For fold 10
    data_train_10_zh = np.concatenate((folds_zh[9:] + folds_zh[:8]))
    data_val_10_zh = np.concatenate(folds_zh[8:9])
    data_test_10_zh = data_test_zh

    ####################################################################

    n_epochs = 60
    batch_size = 48
    output_dim = 1  # Output dimension for binary classification (1 for binary)
    learning_rate = 0.001
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss

    results = {}
    truth = []
    predictions = []
    test_scores = []
    last_epoch_accuracies = []

    for n_fold in range(1, 11):

        normalized_train_en, normalized_val_en, normalized_test_en, y_train_en, y_val_en = normalize_and_split_no_test(
            eval(f"data_train_{n_fold}_en"), eval(f"data_val_{n_fold}_en"), eval(f"data_test_{n_fold}_en"))
        normalized_train_zh, normalized_val_zh, normalized_test_zh, y_train_zh, y_val_zh = normalize_and_split_no_test(
            eval(f"data_train_{n_fold}_zh"), eval(f"data_val_{n_fold}_zh"), eval(f"data_test_{n_fold}_zh"))

        Xtrain = np.concatenate([normalized_train_en, normalized_train_zh], axis=0)
        y_train = np.concatenate([y_train_en, y_train_zh], axis=0)
        Xval = np.concatenate([normalized_val_en, normalized_val_zh], axis=0)
        y_val = np.concatenate([y_val_en, y_val_zh], axis=0)
        Xtest = np.concatenate([normalized_test_en, normalized_test_zh], axis=0)

        input_dim = Xtrain.shape[1]
        model = SingleLayerClassifier(input_dim, output_dim)
        model.apply(reset_weights)
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
        batches_per_epoch = len(Xtrain) // batch_size

        best_val_loss = float('inf')
        patience = 6
        num_epochs_no_improve = 0
        best_accuracy = 0.0

        for epoch in range(n_epochs):
            model.train()
            for i in range(batches_per_epoch):
                optimizer.zero_grad()
                start = i * batch_size
                end = start + batch_size
                Xbatch = Xtrain[start:end]
                ybatch = y_train[start:end]
                Xbatch = torch.tensor(Xbatch, dtype=torch.float32)
                ybatch = torch.tensor(ybatch, dtype=torch.float32)
                y_pred = model(Xbatch)
                loss = criterion(y_pred.flatten(), ybatch)
                loss.backward()
                optimizer.step()

            # Evaluation on the validation set
            model.eval()
            with torch.no_grad():
                Xval_tensor = torch.tensor(Xval, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
                y_val_pred = model(Xval_tensor)
                val_loss = criterion(y_val_pred.flatten(), y_val_tensor)
                accuracy_val = (y_val_pred.round() == y_val_tensor).float().mean()

            if val_loss < best_val_loss and accuracy_val > best_accuracy:
                best_val_loss = val_loss
                best_accuracy = accuracy_val
                num_epochs_no_improve = 0
                if epoch == n_epochs - 1:
                    last_epoch_accuracy = accuracy_val
            else:
                num_epochs_no_improve += 1
                if num_epochs_no_improve >= patience:
                    last_epoch_accuracies.append(accuracy_val)
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Evaluation on the test set
        model.eval()
        with torch.no_grad():
            Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32)
            #  y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
            y_pred = model(Xtest_tensor)
        # accuracy = (y_pred.round() == y_test_tensor).float().mean()
        test_scores.append(y_pred.detach().numpy())

    average_val_acc = np.mean(last_epoch_accuracies)
    print(f"Average validation accuracy for {feat_name}: {average_val_acc}")
    predictions_all, scores = predict_average_scores(list(test_scores))
    all_sps_test = speaker_id_en + speaker_id_zh
    dict = {'names': all_sps_test, 'predictions': predictions_all,
            'score': scores, 'accuracy_validation': round(average_val_acc, 2)}

    df2 = pd.DataFrame(dict)
    df2 = df2.sort_values(by=['names'])
    out_path_score_file = os.path.join(out_path_scores, f"{feat_name}.csv")
    df2.to_csv(out_path_score_file)
