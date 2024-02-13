out_rf =  ''
out_xg =  ''
out_mlp = ''
out_svm = ''
out_bagg  = ''
chinese_sps = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_speaker_division_helin/zh.json'
english_sps = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_speaker_division_helin/en.json'
lang_id = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/lang_id_train/lang_ids.csv'
feats_names = ['trillsson']

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

def get_n_folds(arrayOfSpeaker):
    data = list(arrayOfSpeaker)  # list(range(len(arrayOfSpeaker)))
    num_of_folds = 10
    n_folds = []
    for i in range(num_of_folds):
        n_folds.append(data[int(i * len(data) / num_of_folds):int((i + 1) * len(data) / num_of_folds)])
    return n_folds


def create_fold_lang(path_dict):
    n_folds = []
    read_dict = json.load(open(path_dict))
    for key, values in read_dict.items():
        fold = list((read_dict[key]).keys())
        n_folds.append(list(set(['taukdial-' + sp.split('-')[1] for sp in fold])))

    return n_folds


chinese_sps_folds = create_fold_lang(chinese_sps)
english_sps_folds = create_fold_lang(english_sps)
labels_df= pd.read_csv('/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_labels/groundtruth.csv')

for feat_name in feats_names:
    print(f"Experiments with {feat_name}")
    feat_pth_pd = f'/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/embeddings/{feat_name}/'
    out_path = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_classification/non_interpretable/'
    out_path_scores = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/non_interpretable/classification/'
    print(f"The output directory exists--> {os.path.isdir(out_path)}")
   # test_only = 0
    # check
    path_files = sorted([os.path.join(feat_pth_pd, elem) for elem in sorted(os.listdir(feat_pth_pd))])
    names = sorted(['taukdial-' + os.path.basename(elem).rsplit('-', -1)[1] for elem in path_files])
    ids = sorted([os.path.basename(elem).rsplit('.npy')[0] + '.wav' for elem in path_files])

    if labels_df['tkdname'].tolist() == ids:
        labels = labels_df['dx'].tolist()
        labels = [1 if elem =='NC' else 0 for elem in labels]
        print('DONE')
    else:
        print('error in the labels order')
    #labels_pd = [0]*len(path_files_pd)
    df_pd = pd.DataFrame(list(zip(names, path_files, labels)), columns = ['names', 'path_feat', 'labels'])

    folds_china = []
    folds_names_china = []
    for i in chinese_sps_folds:
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
            names.append(os.path.basename(path).split('.npy')[0])
        folds_china.append(data_fold)
        folds_names_china.append(names)

    folds_eng = []
    folds_names_eng = []
    for i in english_sps_folds:
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
            names.append(os.path.basename(path).split('.npy')[0])
        folds_eng.append(data_fold)
        folds_names_eng.append(names)

    data_train_1_china = np.concatenate(folds_china[:9])
    data_test_1_china = np.concatenate(folds_china[-1:])
    data_train_2_china = np.concatenate(folds_china[1:])
    data_test_2_china = np.concatenate(folds_china[:1])
    data_train_3_china = np.concatenate(folds_china[2:] + folds_china[:1])
    data_test_3_china = np.concatenate(folds_china[1:2])
    data_train_4_china = np.concatenate(folds_china[3:] + folds_china[:2])
    data_test_4_china = np.concatenate(folds_china[2:3])
    data_train_5_china = np.concatenate(folds_china[4:] + folds_china[:3])
    data_test_5_china = np.concatenate(folds_china[3:4])
    data_train_6_china = np.concatenate(folds_china[5:] + folds_china[:4])
    data_test_6_china = np.concatenate(folds_china[4:5])
    data_train_7_china = np.concatenate(folds_china[6:] + folds_china[:5])
    data_test_7_china = np.concatenate(folds_china[5:6])
    data_train_8_china = np.concatenate(folds_china[7:] + folds_china[:6])
    data_test_8_china = np.concatenate(folds_china[6:7])
    data_train_9_china = np.concatenate(folds_china[8:] + folds_china[:7])
    data_test_9_china = np.concatenate(folds_china[7:8])
    data_train_10_china = np.concatenate(folds_china[9:] + folds_china[:8])
    data_test_10_china = np.concatenate(folds_china[8:9])

    data_test_1_names_china = np.concatenate(folds_names_china[-1:])
    data_test_2_names_china = np.concatenate(folds_names_china[:1])
    data_test_3_names_china = np.concatenate(folds_names_china[1:2])
    data_test_4_names_china = np.concatenate(folds_names_china[2:3])
    data_test_5_names_china = np.concatenate(folds_names_china[3:4])
    data_test_6_names_china = np.concatenate(folds_names_china[4:5])
    data_test_7_names_china = np.concatenate(folds_names_china[5:6])
    data_test_8_names_china = np.concatenate(folds_names_china[6:7])
    data_test_9_names_china = np.concatenate(folds_names_china[7:8])
    data_test_10_names_china = np.concatenate(folds_names_china[8:9])

    data_train_1_eng = np.concatenate(folds_eng[:9])
    data_test_1_eng = np.concatenate(folds_eng[-1:])
    data_train_2_eng = np.concatenate(folds_eng[1:])
    data_test_2_eng = np.concatenate(folds_eng[:1])
    data_train_3_eng = np.concatenate(folds_eng[2:] + folds_eng[:1])
    data_test_3_eng = np.concatenate(folds_eng[1:2])
    data_train_4_eng = np.concatenate(folds_eng[3:] + folds_eng[:2])
    data_test_4_eng = np.concatenate(folds_eng[2:3])
    data_train_5_eng = np.concatenate(folds_eng[4:] + folds_eng[:3])
    data_test_5_eng = np.concatenate(folds_eng[3:4])
    data_train_6_eng = np.concatenate(folds_eng[5:] + folds_eng[:4])
    data_test_6_eng = np.concatenate(folds_eng[4:5])
    data_train_7_eng = np.concatenate(folds_eng[6:] + folds_eng[:5])
    data_test_7_eng = np.concatenate(folds_eng[5:6])
    data_train_8_eng = np.concatenate(folds_eng[7:] + folds_eng[:6])
    data_test_8_eng = np.concatenate(folds_eng[6:7])
    data_train_9_eng = np.concatenate(folds_eng[8:] + folds_eng[:7])
    data_test_9_eng = np.concatenate(folds_eng[7:8])
    data_train_10_eng = np.concatenate(folds_eng[9:] + folds_eng[:8])
    data_test_10_eng = np.concatenate(folds_eng[8:9])

    data_test_1_names_eng = np.concatenate(folds_names_eng[-1:])
    data_test_2_names_eng = np.concatenate(folds_names_eng[:1])
    data_test_3_names_eng = np.concatenate(folds_names_eng[1:2])
    data_test_4_names_eng = np.concatenate(folds_names_eng[2:3])
    data_test_5_names_eng = np.concatenate(folds_names_eng[3:4])
    data_test_6_names_eng = np.concatenate(folds_names_eng[4:5])
    data_test_7_names_eng = np.concatenate(folds_names_eng[5:6])
    data_test_8_names_eng = np.concatenate(folds_names_eng[6:7])
    data_test_9_names_eng = np.concatenate(folds_names_eng[7:8])
    data_test_10_names_eng = np.concatenate(folds_names_eng[8:9])

    ##   inner chinese
    if test_only == 0:
        best_params = []
        for i in range(1, 11):
            print(i)
            normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}_china"),
                                                                               eval(f"data_test_{i}_china"))
            # %
            tuned_params = {"PCA_n": [30, 40, 50, 70, 100, 150, 160, 170, 200, 290]}  # per speaker
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
        normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}_china"),
                                                                           eval(f"data_test_{i}_china"))
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
    #report = classification_report(truth, predictions, output_dict=True)
    #df = pd.DataFrame(report).transpose()
    #df['best_PCA_param'] = best_param
    #df['AUROC'] = roc_auc_score(truth, test_scores)
    #df['sensitivity'] = sensitivity
    #df['specificity'] = specificity
   ##file_out = os.path.join(out_path, feat_name + "_" + "PCA_results.csv")
   #df.to_csv(file_out)
   ##
   #all_names = list(data_test_1_names) + list(data_test_2_names) + list(data_test_3_names) \
   #            + list(data_test_4_names) + list(data_test_5_names) + list(data_test_6_names) \
   #            + list(data_test_7_names) + list(data_test_8_names) + list(data_test_9_names) \
   #            + list(data_test_10_names)
   #print(all_names)

   #dict = {'names': all_names, 'truth': truth, 'predictions': predictions, 'score': test_scores}
   #df2 = pd.DataFrame(dict)
   #file_out2 = os.path.join(out_path_scores, feat_name + '.csv')
   #df2.to_csv(file_out2)

    ##   ####################### ENGLISH #################################
    if test_only == 0:
        best_params = []
        for i in range(1, 11):
            print(i)
            normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}_eng"),
                                                                               eval(f"data_test_{i}_eng"))
            # %
            tuned_params = {"PCA_n": [30, 40, 50, 70, 100, 150, 160, 170, 200, 290]}  # per speaker
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

##############################################################################################

    thresholds = []
    predictions = []
    truth = []
    test_scores = []
    for i in range(1, 11):
        print(i)
        normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}_eng"),
                                                                           eval(f"data_test_{i}_eng"))
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
   # report = classification_report(truth, predictions, output_dict=True)
   # df = pd.DataFrame(report).transpose()
   # df['best_PCA_param'] = best_param
   # df['AUROC'] = roc_auc_score(truth, test_scores)
   # df['sensitivity'] = sensitivity
   # df['specificity'] = specificity
    #file_out = os.path.join(out_path, feat_name + "_" + "PCA_results.csv")
    #df.to_csv(file_out)
    ##
    #all_names = list(data_test_1_names) + list(data_test_2_names) + list(data_test_3_names) \
    #            + list(data_test_4_names) + list(data_test_5_names) + list(data_test_6_names) \
    #            + list(data_test_7_names) + list(data_test_8_names) + list(data_test_9_names) \
    #            + list(data_test_10_names)
    #print(all_names)
#
    #dict = {'names': all_names, 'truth': truth, 'predictions': predictions, 'score': test_scores}
    #df2 = pd.DataFrame(dict)
    #file_out2 = os.path.join(out_path_scores, feat_name + '.csv')
    #df2.to_csv(file_out2)
#