# coding: utf-8
import sys
sys.path.append("/export/b16/afavaro/TAUKADIAL-2024/")
from classification_experiments.PCA_PLDA_EER_Classifier import PCA_PLDA_EER_Classifier
import random
import pandas as pd
import numpy as np
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

feats_names = ['xvector', 'trillsson', 'clip_truncation', 'sentence_tr_multiling'] #'whisper']

labels_df= pd.read_csv('/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_labels/groundtruth.csv')
lang_id = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/lang_id_train/lang_ids.csv'

for feat_name in feats_names:
    print(f"Experiments with {feat_name}")
    feat_pth_pd = f'/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/embeddings/{feat_name}/'
    out_path = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/'
    print(f"The output directory exists--> {os.path.isdir(out_path)}")
   # test_only = 0

    path_files = [os.path.join(feat_pth_pd, elem) for elem in sorted(os.listdir(feat_pth_pd))]
    names = ['taukdial-' + os.path.basename(elem).rsplit('-', -1)[1] for elem in path_files]
    ids = [os.path.basename(elem).rsplit('.npy')[0] + '.wav' for elem in path_files]
    if labels_df['tkdname'].tolist() == ids:
        labels = labels_df['dx'].tolist()
        labels = [1 if elem =='NC' else 0 for elem in labels]
        print('DONE')
    else:
        print('error in the labels order')
    #labels_pd = [0]*len(path_files_pd)
    df_pd = pd.DataFrame(list(zip(names, path_files, labels)), columns =['names', 'path_feat', 'labels'])
#
    arrayOfSpeaker_cn = sorted(list(set(df_pd.groupby('labels').get_group(1)['names'].tolist())))
    random.Random(random_seed).shuffle(arrayOfSpeaker_cn)
    ##
    arrayOfSpeaker_pd =  sorted(list(set(df_pd.groupby('labels').get_group(0)['names'].tolist())))
    random.Random(random_seed).shuffle(arrayOfSpeaker_pd)

    print(arrayOfSpeaker_pd)
    print(arrayOfSpeaker_cn)
   ## print('experimental groups length')
  # # print(len(spain.groupby('labels').get_group(1)))
  # # print(len(spain.groupby('labels').get_group(0)))
  # # print(len(arrayOfSpeaker_pd), len(arrayOfSpeaker_cn))
    cn_sps = get_n_folds(arrayOfSpeaker_cn)
    pd_sps = get_n_folds(arrayOfSpeaker_pd)
#
    data = []
    for cn_sp, pd_sp in zip(sorted(cn_sps, key=len), sorted(pd_sps, key=len, reverse=True)):
        data.append(cn_sp + pd_sp)
    n_folds = sorted(data, key=len, reverse=True)
    #print(n_folds)
#
    #folds = []
    #for fold in n_folds:
    #    names_all = []
    #    data_fold = np.array(())  # %
    #    data_i = df_pd[df_pd["names"].isin(fold)]
    #    names = list(set(data_i['names'].tolist()))
    #    for name in names:
    #        names_all.append(name)
    #        gr_sp = data_i.groupby('names').get_group(name)
    #        label_row = (gr_sp['labels'].tolist())[0]
    #        read_vect = [np.load(vec) for vec in gr_sp['path_feat'].tolist()]
    #        read_vect = [vec / np.linalg.norm(vec) for vec in read_vect]
    #        feat = np.mean(read_vect, axis=0)
    #        feat = feat / np.linalg.norm(feat)
    #        feat = np.append(feat, label_row)
    #        data_fold = np.vstack((data_fold, feat)) if data_fold.size else feat
    #    folds.append(data_fold)

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
#
#
    ##      # inner folds cross-validation - hyperparameter search
  # # #if test_only == 0:
  # # #    best_params = []
  # # #    for i in range(1, 11):
  # # #        print(i)
  # # #        normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}"),
  # # #                                                                           eval(f"data_test_{i}"))
  # # #        # %
  # # #        tuned_params = {"PCA_n": [100, 200, 300, 400, 500]}
  # # #        #tuned_params = {"PCA_n": [500]}
  # # #        model = PCA_PLDA_EER_Classifier(normalize=0)
  # # #        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
  # # #        grid_search = GridSearchCV(estimator=model, param_grid=tuned_params, n_jobs=-1, cv=cv, scoring='accuracy',
  # # #                                   error_score=0)
  # # #        grid_result = grid_search.fit(normalized_train_X, y_train)
  # # #        # summarize result
  # # #        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  # # #        print(grid_result.best_params_)
  # # #        means = grid_result.cv_results_['mean_test_score']
  # # #        stds = grid_result.cv_results_['std_test_score']
  # # #        params = grid_result.cv_results_['params']
  # # #        print(means)
  # # #        best_params.append(grid_result.best_params_['PCA_n'])
  # # #    # get best params
  # # #    print('**********best pca n:')
  # # #    best_param = mode(best_params)
  # # #    print(best_param)
#
    if feat_name == "whisper":
      best_param=40
    else:
        best_param = 40
    # outer folds testing
    thresholds = []
    predictions = []
    truth = []
    test_scores = []
    for i in range(1, 11):
        print(i)
        normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}"), eval(f"data_test_{i}"))
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
        thresholds = thresholds + [model.eer_threshold]*len(y_test)
       # print(len(y_test))
       # print(len(thresholds))
       # print(thresholds[:10])
        #print(test_scores)
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