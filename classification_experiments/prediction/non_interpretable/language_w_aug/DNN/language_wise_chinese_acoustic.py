# ['wav2vec_128', 'wav2vec_53',  'trillsson', 'xvector']
#feats_names = ['XLM-Roberta-Large-Vit-L-14', 'lealla-base',
#               'multilingual-e5-large',
#               'text2vec-base-multilingual', 'xlm-roberta-base',
#               'distiluse-base-multilingual-cased',
#               'distiluse-base-multilingual-cased-v1', 'bert-base-multilingual-cased',
#               'LaBSE', 'trillsson', 'xvector']

feats_names = ['trillsson', 'xvector']

#
lang_id = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/lang_id_train/lang_ids.csv'
path_labels = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_labels/groundtruth.csv'

feat_pths = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/embeddings/'
feat_pths_augmented = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats_augmented/embeddings_english/'

out_path = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_augmented/results_per_language/chinese/prediction/non_interpretable_sigmoid/'
out_path_scores = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language_aug/chinese/non_interpretable_sigmoid/prediction/'
list_sps = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_speaker_division_helin/zh.json'

import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score

seed = 40
torch.manual_seed(seed)


def normalize_and_split(train_split, val_split, test_split):
    train_set = train_split
    test_set = test_split
    val_set = val_split

    feat_train = train_set[:, :-2]
    lab_train = train_set[:, -2:-1]
    lab_train = lab_train.astype('int')

    feat_val = val_split[:, :-2]
    lab_val = val_split[:, -2:-1]
    lab_val = lab_val.astype('int')

    feat_test = test_set[:, :-2]
    lab_test = test_set[:, -2:-1]
    lab_test = lab_test.astype('int')

    # X = StandardScaler().fit_transform(matrix_feat)

    X_train, X_val, X_test, y_train, y_val, y_test = feat_train, feat_val, feat_test, lab_train, lab_val, lab_test
    y_test = y_test.ravel()
    y_train = y_train.ravel()
    y_val = y_val.ravel()
    X_train = X_train.astype('float')
    X_test = X_test.astype('float')
    normalized_test_X = (X_test - X_train.mean(0)) / (X_train.std(0) + 0.01)
    normalized_train_X = (X_train - X_train.mean(0)) / (X_train.std(0) + 0.01)
    normalized_val_X = (X_val - X_train.mean(0)) / (X_train.std(0) + 0.01)

    return normalized_train_X, normalized_val_X, normalized_test_X, y_train, y_val, y_test


class SingleLayerClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleLayerClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x.squeeze(1)


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


for feat_name in feats_names:
    print(f"Experiments with {feat_name}")
    n_folds_names = []
    n_folds_data = []
    all_folds_info = []

    read_dict = json.load(open(list_sps))
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

    folds = []
    for fold in all_folds_info:
        data_fold = np.array(())  # %
        for speaker in fold:
            label_row = speaker[-2]
            mmse = speaker[-1]
            feat = np.load(speaker[0])
            feat = np.append(feat, label_row)
            feat = np.append(feat, mmse)
            data_fold = np.vstack((data_fold, feat)) if data_fold.size else feat
        folds.append(data_fold)

    folds_augmented = []
    for fold in all_folds_info:
        data_fold = np.array(())  # %
        for speaker in fold:
            speaker_name = os.path.basename(speaker[0]).split('.npy')[0]
            # print(f'here {speaker_name}')
            label_row = speaker[-2]
            mmse = speaker[-1]
            if feat_name ==  'trillsson' or feat_name== 'xvector' or feat_name == 'wav2vec_128' or feat_name == 'wav2vec_53':
                all_copies = [0, 1, 2, 3] #3 is the best x spech
            else:
                all_copies = [0, 1, 6]#np.arange(0, 7)
           # print(all_copies, feat_name)
            all_augmented_copies = [os.path.join(feat_pths_augmented, feat_name, speaker_name +f'-{num}.npy')  for num in all_copies]

            for copy in all_augmented_copies:

                feat = np.load(copy)
                feat = np.append(feat, label_row)
                feat = np.append(feat, mmse)
                data_fold = np.vstack((data_fold, feat)) if data_fold.size else feat
        folds_augmented.append(data_fold)

    data_train_1 = np.concatenate(folds_augmented[:8])
    data_val_1 = np.concatenate(folds[8:9])
    data_test_1 = np.concatenate(folds[9:])

    # For fold 2
    data_train_2 = np.concatenate((folds_augmented[1:-1]))
    data_val_2 = np.concatenate(folds[-1:])
    data_test_2 = np.concatenate(folds[:1])

    # For fold 3
    data_train_3 = np.concatenate(folds_augmented[2:])
    data_val_3 = np.concatenate(folds[:1])
    data_test_3 = np.concatenate(folds[1:2])

    # For fold 4
    data_train_4 = np.concatenate((folds_augmented[3:] + folds_augmented[:1]))
    data_val_4 = np.concatenate(folds[1:2])
    data_test_4 = np.concatenate(folds[2:3])
    # For fold 5
    data_train_5 = np.concatenate((folds_augmented[4:] + folds_augmented[:2]))
    data_val_5 = np.concatenate(folds[2:3])
    data_test_5 = np.concatenate(folds[3:4])

    # For fold 6
    data_train_6 = np.concatenate((folds_augmented[5:] + folds_augmented[:3]))
    data_val_6 = np.concatenate(folds[3:4])
    data_test_6 = np.concatenate(folds[4:5])
    # For fold 7
    data_train_7 = np.concatenate((folds_augmented[6:] + folds_augmented[:4]))
    data_val_7 = np.concatenate(folds[4:5])
    data_test_7 = np.concatenate(folds[5:6])

    # For fold 8
    data_train_8 = np.concatenate((folds_augmented[7:] + folds_augmented[:5]))
    data_val_8 = np.concatenate(folds[5:6])
    data_test_8 = np.concatenate(folds[6:7])

    # For fold 9
    data_train_9 = np.concatenate((folds_augmented[8:] + folds_augmented[:6]))
    data_val_9 = np.concatenate(folds[6:7])
    data_test_9 = np.concatenate(folds[7:8])

    # For fold 10
    data_train_10 = np.concatenate((folds_augmented[9:] + folds_augmented[:7]))
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

    n_epochs = 40
    batch_size = 48
    input_dim = data_train_1.shape[1] - 2  # Subtract 1 for the label column and 1 for mmse
    # hidden_dim = 40  # Hidden dimension of the fully connected layer
    output_dim = 1  # Output dimension for binary classification (1 for binary)
    learning_rate = 0.001
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss

    results = {}
    test_scores = []
    truth = []
    predictions = []

    for n_fold in range(1, 11):
        # print(n_fold)
        model = SingleLayerClassifier(input_dim, output_dim)
        # model = BinaryClassifier(input_dim, input_dim)
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # DATA
        Xtrain, Xval, Xtest, y_train, y_val, y_test = normalize_and_split(
            eval(f"data_train_{n_fold}"), eval(f"data_val_{n_fold}"), eval(f"data_test_{n_fold}"))

        batches_per_epoch = len(Xtrain) // batch_size

        best_val_loss = float('inf')
        patience = 5
        num_epochs_no_improve = 0
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

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                num_epochs_no_improve = 0
            else:
                num_epochs_no_improve += 1
                if num_epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Evaluation on the test set
        model.eval()
        with torch.no_grad():
            Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
            y_pred = model(Xtest_tensor)
        accuracy = (y_pred.round() == y_test_tensor).float().mean()
        test_scores.append(y_pred.detach().numpy())
        predictions.append(y_pred.round().detach().numpy())
        truth.append(y_test_tensor.detach().numpy())
        results[n_fold] = accuracy

    test_scores = np.concatenate(test_scores)
    truth = np.concatenate(truth)
    predictions = np.concatenate(predictions)

    # report
    print()
    print('----------')
    print('----------')
    print("Final results")
    print(classification_report(truth, predictions, output_dict=False))
    print(confusion_matrix(truth, predictions))
    tn, fp, fn, tp = confusion_matrix(truth, predictions).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print('specificity')
    print(specificity)
    print('sensitivity')
    print(sensitivity)
    print('ROC_AUC')
    print(roc_auc_score(truth, test_scores))
    print('*************')
    print('*************')
    report = classification_report(truth, predictions, output_dict=True)

    df = pd.DataFrame(report).transpose()
    df['AUROC'] = roc_auc_score(truth, test_scores)
    df['sensitivity'] = sensitivity
    df['specificity'] = specificity
    # file_out = os.path.join(out_path, feat_name + "_" + ".csv")
    # df.to_csv(file_out)

    ########################################################################################################################

    all_names = list(data_test_1_names) + list(data_test_2_names) + list(data_test_3_names) \
                + list(data_test_4_names) + list(data_test_5_names) + list(data_test_6_names) \
                + list(data_test_7_names) + list(data_test_8_names) + list(data_test_9_names) \
                + list(data_test_10_names)
    print(all_names)

    dict = {'names': all_names, 'truth': truth, 'predictions': predictions, 'score': test_scores}
# df2 = pd.DataFrame(dict)
# file_out2 = os.path.join(out_path_scores, feat_name + '.csv')
# df2.to_csv(file_out2)

