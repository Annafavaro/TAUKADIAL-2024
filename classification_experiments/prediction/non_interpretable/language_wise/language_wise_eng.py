feats_names = ['trillsson', 'xvector', 'wav2vec', 'whisper']
english_sps = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_speaker_division_helin/en.json'
lang_id = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/lang_id_train/lang_ids.csv'
path_labels = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_labels/groundtruth.csv'
feat_pths = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/embeddings/'

out_path_scores ='/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language/english/non_interpretable/prediction/'
out_path = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_per_language/english/prediction/non_interpretable/'

import sys
sys.path.append("/export/b16/afavaro/TAUKADIAL-2024/")
import json
from classification_experiments.PCA_PLDA_EER_Classifier import PCA_PLDA_EER_Classifier
import random
import pandas as pd
import numpy as np
from statistics import mode
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
random_state = 20
random_seed = 20
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
test_only = 0

def normalize(train_split, test_split): ## when prediction
    train_set = train_split
    test_set = test_split

    feat_train = train_set[:, :-2]
    lab_train = train_set[:, -2:-1]
    lab_train = lab_train.astype('int')

    feat_test = test_set[:, :-2]
    lab_test = test_set[:, -2:-1]
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


#def normalize(train_set, test_set):
#
#    feat_train = train_set[:, :-2]
#    lab_train = train_set[:, -2:-1]
#    lab_train = lab_train.astype('int')
#
#    feat_test = test_set[:, :-2]
#    lab_test = test_set[:, -2:-1]
#    lab_test = lab_test.astype('int')
#
#    control_group = train_set[train_set[:, -2] == 1]  # controls
#    control_group = control_group[:, :-2]  # remove labels from features CNs
#
#    median = np.median(control_group, axis=0)
#    std = np.std(control_group, axis=0)
#
#    X_train, X_test, y_train, y_test = feat_train, feat_test, lab_train, lab_test
#    y_test = y_test.ravel()
#    y_train = y_train.ravel()
#    X_train = X_train.astype('float')
#    X_test = X_test.astype('float')
#    normalized_train_X = (X_train - median) / (std + 0.01)
#    normalized_test_X = (X_test - median) / (std + 0.01)
#
#    return normalized_train_X, normalized_test_X, y_train, y_test

def create_fold_lang(path_dict):
    n_folds = []
    read_dict = json.load(open(path_dict))
    for key, values in read_dict.items():
        fold = list((read_dict[key]).keys())
        n_folds.append(list(set(['taukdial-' + sp.split('-')[1] for sp in fold])))

    return n_folds


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

    print('#################### Test China ############################')
    ##   inner chinese
    if test_only == 0:
        best_params = []
        for i in range(1, 11):
            print(i)
            normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}"),
                                                                               eval(f"data_test_{i}"))
            # %
            tuned_params = {"PCA_n": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}  # per speaker
            model = PCA_PLDA_EER_Classifier(normalize=0)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
            grid_search = GridSearchCV(estimator=model, param_grid=tuned_params, n_jobs=-1, cv=cv, scoring='accuracy',
                                       error_score=0)
            grid_result = grid_search.fit(normalized_train_X, y_train)
            # summarize result
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            print(grid_result.best_params_)
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            print(means)
            best_params.append(grid_result.best_params_['PCA_n'])
        # get best params
        print('**********best pca n:')
        best_param = mode(best_params)

    thresholds = []
    predictions = []
    truth = []
    test_scores = []
    for i in range(1, 11):
        print(i)
        normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}"),
                                                                           eval(f"data_test_{i}"))
        y_test = y_test.tolist()
        model = PCA_PLDA_EER_Classifier(PCA_n=best_param, normalize=0)
        model.fit(normalized_train_X, y_train)
        grid_predictions = model.predict(normalized_test_X)
        print(model.eer_threshold)
        grid_test_scores = model.predict_scores_list(normalized_test_X)
        predictions = predictions + grid_predictions
        truth = truth + y_test
        print(classification_report(y_test, grid_predictions, output_dict=False))
        test_scores += grid_test_scores[:, 0].tolist()
        thresholds = thresholds + [model.eer_threshold] * len(y_test)

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
    df['best_PCA_param'] = best_param
    df['AUROC'] = roc_auc_score(truth, test_scores)
    df['sensitivity'] = sensitivity
    df['specificity'] = specificity
    file_out = os.path.join(out_path, feat_name + "_" + "PCA_results.csv")
    df.to_csv(file_out)
    #
    all_names = list(data_test_1_names) + list(data_test_2_names) + list(data_test_3_names) \
                + list(data_test_4_names) + list(data_test_5_names) + list(data_test_6_names) \
                + list(data_test_7_names) + list(data_test_8_names) + list(data_test_9_names) \
                + list(data_test_10_names)
    print(all_names)

    dict = {'names': all_names, 'truth': truth, 'predictions': predictions, 'score': test_scores}
    df2 = pd.DataFrame(dict)
    file_out2 = os.path.join(out_path_scores, feat_name + '.csv')
    df2.to_csv(file_out2)