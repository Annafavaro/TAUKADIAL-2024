out_path_scores = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language/english_multi/prediction/'
out_path = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_per_language/english_multi/prediction/'

feats_names = ['XLM-Roberta-Large-Vit-L-14', 'lealla-base',
               'multilingual-e5-large', 'whisper'
               'text2vec-base-multilingual', 'xlm-roberta-base',
               'distiluse-base-multilingual-cased',
               'distiluse-base-multilingual-cased-v1',
               'bert-base-multilingual-cased', 'LaBSE', 'wav2vec_128',
               'wav2vec_53', 'trillsson', 'xvector']

names_to_keep_cn = [
    # cn2 --> use this group if you want to consider only AD in the analysis.
  #  'AD_002', 'AD_017', 'AD_020', 'NLS_006', 'NLS_073', 'NLS_075', 'NLS_107', 'NLS_111',
  #  'PEC_002', 'PEC_003', 'PEC_006', 'PEC_007', 'PEC_010', 'PEC_011', 'PEC_012', 'PEC_013',
    'PEC_021', 'PEC_024', 'PEC_028', 'PEC_031', 'PEC_032', 'PEC_037', 'PEC_038', 'PEC_040',
    'PEC_042', 'PEC_043', 'PEC_046', 'PEC_047', 'PEC_049', 'PEC_050', 'PEC_059', 'PEC_060',
    'PEC_062']

names_to_keep_ad = ['AD_003', 'AD_004', 'AD_007', 'AD_008','AD_012', 'AD_013',  'AD_014',
                    'AD_015',  'AD_018', 'AD_019', 'AD_021', 'AD_022', 'AD_023', 'AD_024']

names_to_keep = names_to_keep_cn + names_to_keep_ad

lang_id = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/lang_id_train/lang_ids.csv'
path_labels = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_labels/groundtruth.csv'
feat_pths = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/embeddings/'

english_sps = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_speaker_division_helin/en.json'
chinese_sps = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_speaker_division_helin/zh.json'

delaware = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats_other_datasets/Delaware/embeddings/'
lu = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats_other_datasets/Lu/embeddings/'
adr = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats_other_datasets/Adress-M/embeddings/'
pitt = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats_other_datasets/Pitt/embeddings/'
nls = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats_other_datasets/NLS/embeddings/'
china = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats_other_datasets/Chinese/embeddings/'

import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
seed = 19
torch.manual_seed(seed)

def normalize_train_set(train_split):

    train_set = train_split
    feat_train = train_set[:, :-1]
    lab_train = train_set[:, -1:]
    lab_train = lab_train.astype('int')

    control_group = train_set[train_set[:, -1] == 1]
    control_group = control_group[:, :-1]  # remove labels from features CNs
    median = np.median(control_group, axis=0)
    std = np.std(control_group, axis=0)

    X_train, y_train = feat_train, lab_train
    y_train = y_train.ravel()
    X_train = X_train.astype('float')
    normalized_train_X = (X_train - median) / (std + 0.01)

    return normalized_train_X, y_train


def normalize_and_split(train_split, val_split, test_split):

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
    feat_val = val_set[:, :-1]
    lab_val = val_set[:, -1:]
    lab_val = lab_val.astype('int')

    feat_test = test_set[:, :-1]
    lab_test = test_set[:, -1:]
    lab_test = lab_test.astype('int')

    # X = StandardScaler().fit_transform(matrix_feat)

    X_train, X_val, X_test, y_train, y_val, y_test = feat_train, feat_val, feat_test, lab_train, lab_val, lab_test
    y_train = y_train.ravel()
    y_val = y_val.ravel()
    y_test = y_test.ravel()

    X_train = X_train.astype('float')
    X_val = X_val.astype('float')
    X_test = X_test.astype('float')

    normalized_train_X = (X_train - median) / (std + 0.01)
    normalized_val_X = (X_val - median) / (std + 0.01)
    normalized_test_X = (X_test - median) / (std + 0.01)

    return normalized_train_X, normalized_val_X, normalized_test_X, y_train, y_val, y_test


class SingleLayerClassifier(nn.Module):

    def __init__(self, input_size, output_size):
        super(SingleLayerClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x.squeeze(1)


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


train_labels = pd.read_csv('/export/c06/afavaro/DementiaBank/ADReSS-M/ADReSS-M-train/training-groundtruth.csv')
train_labels_adr = train_labels.sort_values(by=['adressfname'])['dx'].tolist()
train_labels_adr = [1 if ids == 'Control' else 0 for ids in train_labels_adr]

for feat_name in feats_names:
    print(f"Experiments with {feat_name}")

    ############# China ###############

    base_dir_china = os.path.join(china, feat_name)
    all_files_china = [os.path.join(base_dir_china, elem) for elem in sorted(os.listdir(base_dir_china))]
    data_fold_china = np.array(())
    for file in all_files_china:
        label_row = os.path.basename(file).split('_')[0]
        label_row = [1 if label_row == 'HC' else 0]
        feat = np.load(file)
        feat = np.append(feat, label_row)
        data_fold_china = np.vstack((data_fold_china, feat)) if data_fold_china.size else feat

    ############# NLS ###############

    base_dir_nls = os.path.join(nls, feat_name)
    all_files_nls = [os.path.join(base_dir_nls, elem) for elem in sorted(os.listdir(base_dir_nls))]
    data_fold_nls = np.array(())
    for file in all_files_nls:
        name = os.path.basename(file).split('_ses')[0]
        if name in names_to_keep:
            if name in names_to_keep_ad:
                label_row = 0
            else:
                label_row = 1
        feat = np.load(file)
        feat = np.append(feat, label_row)
        data_fold_nls = np.vstack((data_fold_nls, feat)) if data_fold_nls.size else feat

    ############# PITT ###############

    base_dir_pitt = os.path.join(pitt, feat_name)
    all_files_pitt = [os.path.join(base_dir_pitt, elem) for elem in sorted(os.listdir(base_dir_pitt))]
    data_fold_pitt = np.array(())
    for file in all_files_pitt:
        #  print(file)
        label_row = os.path.basename(file).split('_')[0]
        label_row = [1 if label_row == 'CN' else 0]
        feat = np.load(file)
        feat = np.append(feat, label_row)
        data_fold_pitt = np.vstack((data_fold_pitt, feat)) if data_fold_pitt.size else feat

    ############# ADRESS-M ###############

    base_dir_adr = os.path.join(adr, feat_name)
    all_files_adr = [os.path.join(base_dir_adr, elem) for elem in sorted(os.listdir(base_dir_adr))]
    data_fold_adr = np.array(())
    for file in zip(all_files_adr, train_labels_adr):
        label_row = file[-1]
        feat = np.load(file[0])
        feat = np.append(feat, label_row)
        data_fold_adr = np.vstack((data_fold_adr, feat)) if data_fold_adr.size else feat

    ############### Lu ###############

    base_dir_lu = os.path.join(lu, feat_name)
    all_files_lu = [os.path.join(base_dir_lu, elem) for elem in os.listdir(base_dir_lu)]
    data_fold_lu = np.array(())
    for file in all_files_lu:
        label_row = os.path.basename(file).split('_')[0]
        label_row = [1 if label_row == 'CN' else 0]
        feat = np.load(file)
        feat = np.append(feat, label_row)
        data_fold_lu = np.vstack((data_fold_lu, feat)) if data_fold_lu.size else feat

    ############### Delaware ###############

    base_dir_del = os.path.join(delaware, feat_name)
    all_files_del = [os.path.join(base_dir_del, elem) for elem in os.listdir(base_dir_del)]
    data_fold_del = np.array(())
    for file in all_files_del:
        #  print(file)
        label_row = os.path.basename(file).split('_')[0]
        label_row = [1 if label_row == 'CN' else 0]
        feat = np.load(file)
        feat = np.append(feat, label_row)
        data_fold_del = np.vstack((data_fold_del, feat)) if data_fold_del.size else feat

    ####################################################################

    n_folds_names = []
    all_folds_info = []

    read_dict = json.load(open(english_sps))
    for key, values in read_dict.items():
        fold_info_general = []
        fold = list((read_dict[key]).keys())
        n_folds_names.append(list([os.path.basename(sp) for sp in fold]))
        fold_info = read_dict[key]  # get data for
        for sp in fold_info:
            fold_info_general.append(
                [os.path.join(feat_pths, feat_name, sp.split('.wav')[0] + '.npy'), (fold_info[sp])['label']]) #append path and labels
        all_folds_info.append(fold_info_general)

    print(n_folds_names[0])
    folds = []
    for fold in all_folds_info:
        data_fold = np.array(())  # %
        for speaker in fold:
            label_row = speaker[-1]
            feat = np.load(speaker[0])
            feat = np.append(feat, label_row)
            data_fold = np.vstack((data_fold, feat)) if data_fold.size else feat
        folds.append(data_fold)

    # For fold 1
    data_train_1_en = np.concatenate(folds[:8])
    data_val_1_en = np.concatenate(folds[8:9])
    data_test_1_en = np.concatenate(folds[9:])

    # For fold 2
    data_train_2_en = np.concatenate((folds[1:-1]))
    data_val_2_en = np.concatenate(folds[-1:])
    data_test_2_en = np.concatenate(folds[:1])

    # For fold 3
    data_train_3_en = np.concatenate(folds[2:])
    data_val_3_en = np.concatenate(folds[:1])
    data_test_3_en = np.concatenate(folds[1:2])

    # For fold 4
    data_train_4_en = np.concatenate((folds[3:] + folds[:1]))
    data_val_4_en = np.concatenate(folds[1:2])
    data_test_4_en = np.concatenate(folds[2:3])
    # For fold 5
    data_train_5_en = np.concatenate((folds[4:] + folds[:2]))
    data_val_5_en = np.concatenate(folds[2:3])
    data_test_5_en = np.concatenate(folds[3:4])

    # For fold 6
    data_train_6_en = np.concatenate((folds[5:] + folds[:3]))
    data_val_6_en = np.concatenate(folds[3:4])
    data_test_6_en = np.concatenate(folds[4:5])
    # For fold 7
    data_train_7_en = np.concatenate((folds[6:] + folds[:4]))
    data_val_7_en = np.concatenate(folds[4:5])
    data_test_7_en = np.concatenate(folds[5:6])

    # For fold 8
    data_train_8_en = np.concatenate((folds[7:] + folds[:5]))
    data_val_8_en = np.concatenate(folds[5:6])
    data_test_8_en = np.concatenate(folds[6:7])

    # For fold 9
    data_train_9_en = np.concatenate((folds[8:] + folds[:6]))
    data_val_9_en = np.concatenate(folds[6:7])
    data_test_9_en = np.concatenate(folds[7:8])

    # For fold 10
    data_train_10_en = np.concatenate((folds[9:] + folds[:7]))
    data_val_10_en = np.concatenate(folds[7:8])
    data_test_10_en = np.concatenate(folds[8:9])

    data_test_1_names_en = np.concatenate(n_folds_names[-1:])
    data_test_2_names_en = np.concatenate(n_folds_names[:1])
    data_test_3_names_en = np.concatenate(n_folds_names[1:2])
    data_test_4_names_en = np.concatenate(n_folds_names[2:3])
    data_test_5_names_en = np.concatenate(n_folds_names[3:4])
    data_test_6_names_en = np.concatenate(n_folds_names[4:5])
    data_test_7_names_en = np.concatenate(n_folds_names[5:6])
    data_test_8_names_en = np.concatenate(n_folds_names[6:7])
    data_test_9_names_en = np.concatenate(n_folds_names[7:8])
    data_test_10_names_en = np.concatenate(n_folds_names[8:9])

    n_epochs = 30
    batch_size = 48
      # Subtract 1 for the label column and 1 for mmse
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

        # DATA
        normalized_train_en, normalized_val_en, normalized_test_en, y_train_en, y_val_en, y_test_en = normalize_and_split(
            eval(f"data_train_{n_fold}_en"), eval(f"data_val_{n_fold}_en"), eval(f"data_test_{n_fold}_en"))

        normalized_train_del, y_train_del = normalize_train_set(data_fold_del)
        normalized_train_lu, y_train_lu = normalize_train_set(data_fold_lu)
        normalized_train_adr, y_train_adr = normalize_train_set(data_fold_adr)
        normalized_train_pitt, y_train_pitt = normalize_train_set(data_fold_pitt)
        normalized_train_nls, y_train_nls = normalize_train_set(data_fold_nls)
        normalized_train_china, y_train_china = normalize_train_set(data_fold_china)

        Xtrain = np.concatenate([normalized_train_nls, normalized_train_en, normalized_train_lu, normalized_train_del, normalized_train_adr ], axis=0)
        y_train = np.concatenate([y_train_nls, y_train_en, y_train_lu, y_train_del, y_train_adr], axis=0)

        Xval = np.concatenate([normalized_val_en], axis=0)
        y_val = np.concatenate([y_val_en], axis=0)

        Xtest = np.concatenate([normalized_test_en], axis=0)
        y_test = np.concatenate([y_test_en], axis=0)

        input_dim = Xtrain.shape[1]
        model = SingleLayerClassifier(input_dim, output_dim)
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

    # Print k-fold cross-validation results
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
    file_out = os.path.join(out_path, feat_name + "_" + ".csv")
    df.to_csv(file_out)
    print('saved')

    ########################################################################################################################
    all_names = (list(data_test_1_names_en) +
              list(data_test_2_names_en) +
              list(data_test_3_names_en) +
              list(data_test_4_names_en) +
              list(data_test_5_names_en) +
              list(data_test_6_names_en) +
              list(data_test_7_names_en) +
              list(data_test_8_names_en) +
              list(data_test_9_names_en) +
              list(data_test_10_names_en))
#
    print(all_names)
    dict = {'names': all_names, 'truth': truth, 'predictions': predictions, 'score': test_scores}
    df2 = pd.DataFrame(dict)
    file_out2 = os.path.join(out_path_scores, feat_name + '.csv')
    df2.to_csv(file_out2)
