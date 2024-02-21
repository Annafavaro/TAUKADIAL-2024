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


def normalize_and_split(train_split, test_split):

    train_set = train_split
    test_set = test_split

    feat_train = train_set[:, :-1]
    lab_train = train_set[:, -1:]
    lab_train = lab_train.astype('int')

    control_group = train_set[train_set[:, -1] == 1]
    control_group = control_group[:, :-1]  # remove labels from features CNs
    median = np.median(control_group, axis=0)
    std = np.std(control_group, axis=0)

    feat_test = test_set[:, :-1]
    lab_test = test_set[:, -1:]
    lab_test = lab_test.astype('int')

    # X = StandardScaler().fit_transform(matrix_feat)

    X_train, X_test, y_train, y_test = feat_train, feat_test, lab_train, lab_test
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    X_train = X_train.astype('float')
    X_test = X_test.astype('float')

    normalized_train_X = (X_train - median) / (std + 0.01)
    normalized_test_X = (X_test - median) / (std + 0.01)

    return normalized_train_X, normalized_test_X, y_train, y_test

train_labels = pd.read_csv('/export/c06/afavaro/DementiaBank/ADReSS-M/ADReSS-M-train/training-groundtruth.csv')
train_labels_adr = train_labels.sort_values(by=['adressfname'])['dx'].tolist()
train_labels_adr = [1 if ids == 'Control' else 0 for ids in train_labels_adr]

for feat_name in feats_names:
    print(f"Experiments with {feat_name}")

   # ############# China ###############
#
   # base_dir_china = os.path.join(china, feat_name)
   # all_files_china = [os.path.join(base_dir_china, elem) for elem in sorted(os.listdir(base_dir_china))]
   # data_fold_china = np.array(())
   # for file in all_files_china:
   #     label_row = os.path.basename(file).split('_')[0]
   #     label_row = [1 if label_row == 'HC' else 0]
   #     feat = np.load(file)
   #     feat = np.append(feat, label_row)
   #     data_fold_china = np.vstack((data_fold_china, feat)) if data_fold_china.size else feat

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

  #  base_dir_pitt = os.path.join(pitt, feat_name)
  #  all_files_pitt = [os.path.join(base_dir_pitt, elem) for elem in sorted(os.listdir(base_dir_pitt))]
  #  data_fold_pitt = np.array(())
  #  for file in all_files_pitt:
  #      #  print(file)
  #      label_row = os.path.basename(file).split('_')[0]
  #      label_row = [1 if label_row == 'CN' else 0]
  #      feat = np.load(file)
  #      feat = np.append(feat, label_row)
  #      data_fold_pitt = np.vstack((data_fold_pitt, feat)) if data_fold_pitt.size else feat

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

    print('#################### Test English ############################')
    ##   inner chinese
    if test_only == 0:
        best_params = []
        for n_fold in range(1, 11):
            print(n_fold)

            normalized_train_en, normalized_test_en, y_train_en, y_test_en = normalize_and_split(
                eval(f"data_train_{n_fold}"), eval(f"data_test_{n_fold}"))

            normalized_train_del, y_train_del = normalize_train_set(data_fold_del)
            normalized_train_lu, y_train_lu = normalize_train_set(data_fold_lu)
            normalized_train_adr, y_train_adr = normalize_train_set(data_fold_adr)
          #  normalized_train_pitt, y_train_pitt = normalize_train_set(data_fold_pitt)
            normalized_train_nls, y_train_nls = normalize_train_set(data_fold_nls)
           # normalized_train_china, y_train_china = normalize_train_set(data_fold_china)

            normalized_train_X = np.concatenate(
                [normalized_train_nls, normalized_train_en, normalized_train_lu, normalized_train_del,
                 normalized_train_adr], axis=0)

            y_train = np.concatenate([y_train_nls, y_train_en, y_train_lu, y_train_del, y_train_adr], axis=0)

            normalized_test_X = np.concatenate([normalized_test_en], axis=0)
            y_test = np.concatenate([y_test_en], axis=0)


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
    for n_fold in range(1, 11):
        print(i)
        normalized_train_en, normalized_test_en, y_train_en, y_test_en = normalize_and_split(
            eval(f"data_train_{n_fold}"), eval(f"data_test_{n_fold}"))

        normalized_train_del, y_train_del = normalize_train_set(data_fold_del)
        normalized_train_lu, y_train_lu = normalize_train_set(data_fold_lu)
        normalized_train_adr, y_train_adr = normalize_train_set(data_fold_adr)
        #normalized_train_pitt, y_train_pitt = normalize_train_set(data_fold_pitt)
        normalized_train_nls, y_train_nls = normalize_train_set(data_fold_nls)
        # normalized_train_china, y_train_china = normalize_train_set(data_fold_china)

        normalized_train_X = np.concatenate(
            [normalized_train_nls, normalized_train_en, normalized_train_lu, normalized_train_del,
             normalized_train_adr], axis=0)

        y_train = np.concatenate([y_train_nls, y_train_en, y_train_lu, y_train_del, y_train_adr], axis=0)

        normalized_test_X = np.concatenate([normalized_test_en], axis=0)
        y_test = np.concatenate([y_test_en], axis=0)

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
   # file_out = os.path.join(out_path, feat_name + "_" + "PCA_results.csv")
   # df.to_csv(file_out)
    #
    all_names = list(data_test_1_names) + list(data_test_2_names) + list(data_test_3_names) \
                + list(data_test_4_names) + list(data_test_5_names) + list(data_test_6_names) \
                + list(data_test_7_names) + list(data_test_8_names) + list(data_test_9_names) \
                + list(data_test_10_names)
    print(all_names)

   # dict = {'names': all_names, 'truth': truth, 'predictions': predictions, 'score': test_scores}
   # df2 = pd.DataFrame(dict)
