out_rf =  '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_per_language/english/prediction/interpretable/RF/'
out_xg =  '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_per_language/english/prediction/interpretable/XG/'
out_svm = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_per_language/english/prediction/interpretable/SVM/'
out_bagg = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_per_language/english/prediction/interpretable/BAGG/'

model_out_preds = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language/english/interpretable/prediction/'
english_sps = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_speaker_division_helin/en.json'

import numpy as np
import pandas as pd
import random
import numpy as np
import random
import os
import json
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score


def normalize(train_set, test_set):
    feat_train = train_set[:, :-1]
    lab_train = train_set[:, -1:]
    lab_train = lab_train.astype('int')

    feat_test = test_set[:, :-1]
    lab_test = test_set[:, -1:]
    lab_test = lab_test.astype('int')

    control_group = train_set[train_set[:, -1] == 1]  # controls
    control_group = control_group[:, :-1]  # remove labels from features CNs

    median = np.median(control_group, axis=0)
    std = np.std(control_group, axis=0)

    X_train, X_test, y_train, y_test = feat_train, feat_test, lab_train, lab_test
    y_test = y_test.ravel()
    y_train = y_train.ravel()
    X_train = X_train.astype('float')
    X_test = X_test.astype('float')
    normalized_train_X = (X_train - median) / (std + 0.01)
    normalized_test_X = (X_test - median) / (std + 0.01)

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
        n_folds.append(list([sp.split('.wav')[0] for sp in fold]))

    return n_folds

cols_ling_to_keep =  ['ID',
    't_word', 't_stopword', 't_punct', 't_uword', 't_sent', 't_char', 'a_word_ps',
    'a_char_ps', 'a_char_pw', 'simp_adj_var', 'simp_adp_var', 'simp_adv_var',
    'simp_aux_var', 'simp_cconj_var', 'simp_det_var', 'simp_intj_var', 'simp_noun_var',
    'simp_num_var', 'simp_part_var', 'simp_pron_var', 'simp_propn_var', 'simp_punct_var',
    'simp_sconj_var', 'simp_sym_var', 'simp_verb_var',
   'root_adj_var', 'root_adp_var', 'root_adv_var', 'root_aux_var', 'root_cconj_var',
    'root_det_var', 'root_intj_var', 'root_noun_var', 'root_num_var', 'root_part_var',
    'root_pron_var', 'root_propn_var', 'root_punct_var', 'root_sconj_var',
    'root_sym_var', 'root_verb_var',
    'corr_adj_var', 'corr_adp_var', 'corr_adv_var', 'corr_aux_var', 'corr_cconj_var',
    'corr_det_var', 'corr_intj_var', 'corr_noun_var', 'corr_num_var', 'corr_part_var',
    'corr_pron_var', 'corr_propn_var', 'corr_punct_var', 'corr_sconj_var', 'corr_sym_var',
    'corr_verb_var',
    'simp_ttr', 'root_ttr', 'corr_ttr', 'bilog_ttr', 'uber_ttr', 'simp_ttr_no_lem',
    'root_ttr_no_lem', 'corr_ttr_no_lem', 'bilog_ttr_no_lem', 'uber_ttr_no_lem',
    'n_adj', 'n_adp', 'n_adv', 'n_aux', 'n_cconj', 'n_det', 'n_intj', 'n_noun', 'n_num',
    'n_part', 'n_pron', 'n_propn', 'n_punct', 'n_sconj', 'n_sym', 'n_verb',
     'n_uadj', 'n_uadp', 'n_uadv', 'n_uaux', 'n_ucconj', 'n_udet', 'n_uintj', 'n_unoun',
    'n_unum', 'n_upart', 'n_upron', 'n_upropn', 'n_upunct', 'n_usconj', 'n_usym',
    'n_uverb', 'a_adj_pw', 'a_adp_pw', 'a_adv_pw', 'a_aux_pw', 'a_cconj_pw', 'a_det_pw', 'a_intj_pw',
    'a_noun_pw', 'a_num_pw', 'a_part_pw', 'a_pron_pw', 'a_propn_pw', 'a_punct_pw',
    'a_sconj_pw', 'a_sym_pw', 'a_verb_pw',
    'a_adj_ps', 'a_adp_ps', 'a_adv_ps', 'a_aux_ps', 'a_cconj_ps', 'a_det_ps', 'a_intj_ps',
    'a_noun_ps', 'a_num_ps', 'a_part_ps', 'a_pron_ps', 'a_propn_ps', 'a_punct_ps',
    'a_sconj_ps',  'a_verb_ps',
]

english_sps_read = create_fold_lang(english_sps)
labels = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_labels/groundtruth.csv'
lang_id = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/lang_id_train/lang_ids.csv'

ling1 = pd.read_csv('/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/interpretable/no_diarization/lexical_sem_en.csv')
ling2 =  pd.read_csv('/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/interpretable/no_diarization/lexical_sem_zh.csv')
ling = pd.concat([ling1,ling2], axis=0)

ling = ling[ling.columns[ling.columns.isin(cols_ling_to_keep)]]
#ling = add_lang(ling, lang_id).reset_index(drop=True)
ling['ID'] = [elem + '.wav' for elem in ling['ID'].tolist()]
ling = ling.sort_values('ID').reset_index(drop=True)
ling = add_labels(ling, labels).reset_index(drop=True)

pause = pd.read_csv('/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/interpretable/no_diarization/pause.csv')
#cols_pause_to_remove = [col for col in pause.columns.tolist() if 'VADInt_2' not in col and 'AudioFile' not in col]
#pause = pause.drop(columns=cols_pause_to_remove)
pause = pause.drop(columns=['Unnamed: 0'])
pause = pause.rename(columns={"AudioFile": "ID"})
pause = pause.sort_values('ID').reset_index(drop=True)
pause = add_labels(pause, labels).reset_index(drop=True)

intensity = pd.read_csv('/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/interpretable/no_diarization/intensity_attributes.csv')
intensity['ID'] = [os.path.basename(elem) for elem in intensity['sound_filepath'].tolist()]
intensity = intensity[intensity.columns.drop(list(intensity.filter(regex='jitter')))]
intensity = intensity[intensity.columns.drop(list(intensity.filter(regex='shimmer')))]
intensity = intensity.drop(columns=['Unnamed: 0', 'sound_filepath', 'lfcc', 'mfcc', 'delta_mfcc', 'delta_delta_mfcc' ])
intensity = intensity.sort_values('ID').reset_index(drop=True)
intensity = add_labels(intensity, labels).reset_index(drop=True)

open_smile_feats = pd.read_csv('/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/compare/no_diarization/compare_features.csv')
open_cols = [col for col in open_smile_feats.columns if 'mfcc' not in col and 'ID' not in col] #+  ['ID']
open_smile_feats = open_smile_feats.drop(columns=open_cols)
open_smile_feats = open_smile_feats.sort_values('ID')
open_smile_feats = add_labels(open_smile_feats, labels).reset_index(drop=True)

prosody =  pd.read_csv('/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/interpretable/no_diarization/prosody.csv')
#prosody = prosody.dropna()
prosody = prosody.rename(columns={"id": "ID"})
prosody = prosody.sort_values('ID')
prosody = add_labels(prosody, labels).reset_index(drop=True)

total = pd.concat([open_smile_feats], axis=1)
#total = pd.concat([ling, pause, intensity, prosody, open_smile_feats], axis=1)#.drop_duplicates()
total = total.loc[:,~total.columns.duplicated()]
total['ID_tot'] = [elem.rsplit('.')[0] for elem in total['ID'].tolist()]
total['ID'] = ['taukdial-' + elem.rsplit('-', -1)[1] for elem in total['ID'].tolist()]
total['labels'] = [1 if elem =='NC' else 0 for elem in total['labels'].tolist()]
##
column_to_move = total.pop('ID')
total['ID'] = column_to_move
column_to_move = total.pop('labels')
total['labels'] = column_to_move
column_to_move = total.pop('ID_tot')
total['ID_tot'] = column_to_move

total = total.dropna(axis='columns')


folds_names = []
folds = []
for fold in english_sps_read:
    data_fold = np.array(())
    fold_name = []
    for speaker in fold:
        name = []
        data_i = total[total["ID_tot"]==speaker]
        fold_name.append(data_i['ID_tot'].tolist()[0])
        feat = data_i.drop(columns=['ID', 'ID_tot']).values
        #feat = np.append(data_i)
        data_fold = np.vstack((data_fold, feat)) if data_fold.size else feat
    folds.append(data_fold)
    folds_names.append(fold_name)

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

all_feats = 35
ling_opensmile = 40
ling_max = 40
prosody_max = 40
open_smile_max = 15
int_max = 30
pause_max = 10

class_name = 'SVC'
predictions = []
truth = []
test_scores = []

feat_names = 'prosody_max'
feat_num_sel = open_smile_max

for num_inter in range(1, 11):
    print(num_inter)

    normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{num_inter}"),
                                                                       eval(f"data_test_{num_inter}"))

    clf = ExtraTreesClassifier()
    clf = clf.fit(normalized_train_X, y_train)
    model = SelectFromModel(clf, prefit=True, max_features=feat_num_sel)
    X_train = model.transform(normalized_train_X)
    cols = model.get_support(indices=True)
    X_test = normalized_test_X[:, cols]
    # SVM
    model1 = SVC(C=10, gamma=0.0005, kernel='rbf')
    grid_result = model1.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    predictions.append(grid_predictions)
    truth.append(y_test)
    model2 = SVC(C=1, gamma=0.05, kernel='rbf', probability=True)
    grid_result = model2.fit(X_train, y_train)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    test_scores.append(grid_predictions)
#   out_model_pred = os.path.join(model_out, f'SVM_pred_{num_inter}_.pkl')

test_scores = list(np.concatenate(test_scores))
truth = np.concatenate(truth)
predictions = np.concatenate(predictions)
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
au = (roc_auc_score(truth, test_scores))
# au = (roc_auc_score(truth.astype(bool), test_scores))
print(au)
#######################################################
report = classification_report(truth, predictions, output_dict=True)
# report = classification_report(truth.astype(bool), predictions.astype(bool), output_dict=True)
df = pd.DataFrame(report).transpose()
df['AUROC'] = roc_auc_score(truth, test_scores)
df['sensitivity'] = sensitivity
df['specificity'] = specificity
file_out = os.path.join(out_svm, f"{feat_names}.csv")
df.to_csv(file_out)

all_names = list(data_test_1_names) + list(data_test_2_names) + list(data_test_3_names) \
           + list(data_test_4_names) + list(data_test_5_names) + list(data_test_6_names) \
           + list(data_test_7_names) + list(data_test_8_names) + list(data_test_9_names) \
           + list(data_test_10_names)
dict = {'names': all_names, 'truth': truth, 'predictions': predictions, 'score': test_scores}
df = pd.DataFrame(dict)
df.to_csv(os.path.join(model_out_preds, f'svm_{feat_names}.csv'))

#################################################################################################

class_name = 'RandomForestClassifier'
print(class_name)
predictions = []
truth = []
test_scores = []

for i in range(1, 11):
    print(i)
    normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}"), eval(f"data_test_{i}"))
    clf = ExtraTreesClassifier()
    clf = clf.fit(normalized_train_X, y_train)
    model = SelectFromModel(clf, prefit=True, max_features=feat_num_sel)
    X_train = model.transform(normalized_train_X)
    cols = model.get_support(indices=True)
    X_test = normalized_test_X[:, cols]

    # SVM
    model = RandomForestClassifier(max_features='log2', n_estimators=1000)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    predictions.append(grid_predictions)
    truth.append(y_test)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    test_scores.append(grid_predictions)

test_scores = list(np.concatenate(test_scores))
# print(test_scores)
truth = np.concatenate(truth)
predictions = np.concatenate(predictions)
print(classification_report(truth.astype(bool), predictions.astype(bool), output_dict=False))
print(confusion_matrix(truth.astype(bool), predictions.astype(bool)))
tn, fp, fn, tp = confusion_matrix(truth.astype(bool), predictions.astype(bool)).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print('specificity')
print(specificity)
print('sensitivity')
print(sensitivity)
print('ROC_AUC')
au = (roc_auc_score(truth.astype(bool), test_scores))
print(au)
#######################################################
report = classification_report(truth.astype(bool), predictions.astype(bool), output_dict=True)
df = pd.DataFrame(report).transpose()
df['AUROC'] = roc_auc_score(truth.astype(bool), test_scores)
df['sensitivity'] = sensitivity
df['specificity'] = specificity
file_out = os.path.join(out_rf, f"{feat_names}.csv")
df.to_csv(file_out)

all_names = list(data_test_1_names) + list(data_test_2_names) + list(data_test_3_names) \
            + list(data_test_4_names) + list(data_test_5_names) + list(data_test_6_names) \
            + list(data_test_7_names) + list(data_test_8_names) + list(data_test_9_names) \
            + list(data_test_10_names)

dict = {'names': all_names, 'truth': truth, 'predictions': predictions, 'score': test_scores}
df = pd.DataFrame(dict)
df.to_csv(os.path.join(model_out_preds, f'rf_{feat_names}.csv'))

#################################################################################################

class_name = 'GradientBoostingClassifier'
print(class_name)
predictions = []
truth = []
test_scores = []

for i in range(1, 11):

    print(i)

    normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}"), eval(f"data_test_{i}"))
    clf = ExtraTreesClassifier()
    clf = clf.fit(normalized_train_X, y_train)
    model = SelectFromModel(clf, prefit=True, max_features=feat_num_sel)
    X_train = model.transform(normalized_train_X)
    cols = model.get_support(indices=True)
    X_test = normalized_test_X[:, cols]

    model = GradientBoostingClassifier(learning_rate=0.001, max_depth=9, n_estimators=1000, subsample=0.5)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    predictions.append(grid_predictions)
    truth.append(y_test)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    test_scores.append(grid_predictions)
#print(f'Cross-Corpora Multilingual---> Test on Parkceleb {time_diagnosis}')
test_scores = list(np.concatenate(test_scores))
# print(test_scores)
truth = np.concatenate(truth)
predictions = np.concatenate(predictions)
print(classification_report(truth.astype(bool), predictions.astype(bool), output_dict=False))
print(confusion_matrix(truth.astype(bool), predictions.astype(bool)))
tn, fp, fn, tp = confusion_matrix(truth.astype(bool), predictions.astype(bool)).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print('specificity')
print(specificity)
print('sensitivity')
print(sensitivity)
print('ROC_AUC')
au = (roc_auc_score(truth.astype(bool), test_scores))
print(au)
report = classification_report(truth.astype(bool), predictions.astype(bool), output_dict=True)
df = pd.DataFrame(report).transpose()
df['AUROC'] = roc_auc_score(truth.astype(bool), test_scores)
df['sensitivity'] = sensitivity
df['specificity'] = specificity
file_out = os.path.join(out_xg,   f"{feat_names}.csv")
df.to_csv(file_out)

all_names = list(data_test_1_names) + list(data_test_2_names) + list(data_test_3_names) \
            + list(data_test_4_names) + list(data_test_5_names) + list(data_test_6_names) \
            + list(data_test_7_names) + list(data_test_8_names) + list(data_test_9_names) \
            + list(data_test_10_names)

dict = {'names': all_names, 'truth': truth, 'predictions': predictions, 'score': test_scores}
df = pd.DataFrame(dict)
df.to_csv(os.path.join(model_out_preds, f'xgb_{feat_names}.csv'))

#################################################################################################

predictions = []
truth = []
test_scores = []
class_name = 'BaggingClassifier'
for i in range(1, 11):

    print(i)

    normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}"), eval(f"data_test_{i}"))
    clf = ExtraTreesClassifier()
    clf = clf.fit(normalized_train_X, y_train)
    model = SelectFromModel(clf, prefit=True, max_features=feat_num_sel)
    X_train = model.transform(normalized_train_X)
    cols = model.get_support(indices=True)
    X_test = normalized_test_X[:, cols]

    model = BaggingClassifier(n_estimators=100, max_samples=0.53)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    predictions.append(grid_predictions)
    truth.append(y_test)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    test_scores.append(grid_predictions)
#print(f'Cross-Corpora Multilingual---> Test on Parkceleb {time_diagnosis}')

test_scores = list(np.concatenate(test_scores))
truth = np.concatenate(truth)
predictions = np.concatenate(predictions)
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
au = (roc_auc_score(truth, test_scores))
au = round(au, 2)
print(au)
print('----------')
report = classification_report(truth, predictions, output_dict=True)
df = pd.DataFrame(report).transpose()
df['AUROC'] = roc_auc_score(truth, test_scores)
df['sensitivity'] = sensitivity
file_out = os.path.join(out_bagg,  f"{feat_names}.csv")
df.to_csv(file_out)

all_names = list(data_test_1_names) + list(data_test_2_names) + list(data_test_3_names) \
            + list(data_test_4_names) + list(data_test_5_names) + list(data_test_6_names) \
            + list(data_test_7_names) + list(data_test_8_names) + list(data_test_9_names) \
            + list(data_test_10_names)

dict = {'names': all_names, 'truth': truth, 'predictions': predictions, 'score': test_scores}
df = pd.DataFrame(dict)
df.to_csv(os.path.join(model_out_preds, f'bagg_{feat_names}.csv'))


