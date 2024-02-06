
SVM_OUT_PATH = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_grid_search_classification/SVM/SVM.txt'
MLP_OUT_PATH = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_grid_search_classification/MLP/MLP.txt'
RF_OUT_PATH = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_grid_search_classification/RF/RF.txt'
XG_OUT_PATH = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_grid_search_classification/XG/XG.txt'
BAGG_OUT_PATH = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/results_training/results_grid_search_classification/BAGG/BAGG.txt'


from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import random
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold



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


def add_lang(df, path_labels):
    path_labels_df = pd.read_csv(path_labels)
    label = path_labels_df['lang'].tolist()
    speak = ['taukdial-' + elem.split('-')[1] for elem in path_labels_df['names'].tolist()]
    # print(speak)
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
    df['lang'] = label_new_

    return df



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

    X_train, X_test, y_train, y_test = feat_train, feat_test, lab_train, lab_test
    y_test = y_test.ravel()
    y_train = y_train.ravel()
    X_train = X_train.astype('float')
    X_test = X_test.astype('float')
    normalized_test_X = (X_test - X_train.mean(0)) / (X_train.std(0) + 0.01)
    normalized_train_X = (X_train - X_train.mean(0)) / (X_train.std(0) + 0.01)

    return normalized_train_X, normalized_test_X, y_train, y_test


labels = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_labels/groundtruth.csv'
lang_id = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/lang_id_train/lang_ids.csv'

ling = pd.read_csv('/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/interpretable/no_diarization/lexical_sem.csv')
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
    'a_sconj_ps', 'a_sym_ps', 'a_verb_ps',
]

ling = ling[ling.columns[ling.columns.isin(cols_ling_to_keep)]]
ling['ID'] = [elem + '.wav' for elem in ling['ID'].tolist()]
ling = ling.sort_values('ID').reset_index(drop=True)
ling = add_labels(ling, labels).reset_index(drop=True)

pause = pd.read_csv('/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/interpretable/no_diarization/pause.csv')
cols_pause_to_remove = [col for col in pause.columns.tolist() if 'VADInt_2' not in col and 'AudioFile' not in col]
pause = pause.drop(columns=cols_pause_to_remove)
pause = pause.rename(columns={"AudioFile": "ID"})
pause = pause.sort_values('ID').reset_index(drop=True)
pause = add_labels(pause, labels).reset_index(drop=True)

intensity = pd.read_csv('/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/interpretable/no_diarization/intensity_attributes.csv')
intensity['ID'] = [os.path.basename(elem) for elem in intensity['sound_filepath'].tolist()]
intensity = intensity.drop(columns=['Unnamed: 0', 'sound_filepath', 'lfcc', 'mfcc', 'delta_mfcc', 'delta_delta_mfcc' ])
intensity = intensity.sort_values('ID').reset_index(drop=True)
intensity = add_labels(intensity, labels).reset_index(drop=True)

open_smile_feats = pd.read_csv('/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/feats/compare/no_diarization/compare_features.csv')
open_smile_feats = open_smile_feats.drop(columns=['Unnamed: 0'])
open_smile_feats = open_smile_feats.sort_values('ID')
open_smile_feats = add_labels(open_smile_feats, labels).reset_index(drop=True)

total = pd.concat([ling, pause, open_smile_feats], axis=1)#.drop_duplicates()
#total = pd.concat([ling], axis=1)#.drop_duplicates()
total = total.loc[:,~total.columns.duplicated()]
total['ID'] = ['taukdial-' + elem.rsplit('-', -1)[1] for elem in total['ID'].tolist()]
total['labels'] = [1 if elem =='NC' else 0 for elem in total['labels'].tolist()]
#
column_to_move = total.pop('ID')
total['ID'] = column_to_move
column_to_move = total.pop('labels')
total['labels'] = column_to_move
#total = add_lang(total, lang_id)

gr = total.groupby('labels')
ctrl_ = gr.get_group(1)
pd_ = gr.get_group(0)

arrayOfSpeaker_cn = ctrl_['ID'].unique()
random.shuffle(arrayOfSpeaker_cn)

arrayOfSpeaker_pd = pd_['ID'].unique()
random.shuffle(arrayOfSpeaker_pd)

cn_sps = get_n_folds(arrayOfSpeaker_cn)
pd_sps = get_n_folds(arrayOfSpeaker_pd)


data = []
for cn_sp, pd_sp in zip(sorted(cn_sps, key=len), sorted(pd_sps, key=len, reverse=True)):
    data.append(cn_sp + pd_sp)
n_folds = sorted(data, key=len, reverse=True)


folds_names = []
folds = []
for i in n_folds:
    name = []
    data_i = total[total["ID"].isin(i)]
    folds_names.append(data_i['ID'].tolist())
    data_i = data_i.drop(columns=['ID'])
    folds.append((data_i).to_numpy())


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


################################################################################################################

svm_parameters = {}
mlp_paramters = {}
rf_paramters = {}
xg_paramters = {}
bagg_paramters = {}

for i in range(1, 11):

    print(i)

    normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}"),
                                                                                       eval(f"data_test_{i}"))
    clf = ExtraTreesClassifier()
    clf = clf.fit(normalized_train_X, y_train)
    model = SelectFromModel(clf, prefit=True, max_features=30)
    X_train = model.transform(normalized_train_X)
    cols = model.get_support(indices=True)
    X_test = normalized_test_X[:, cols]
    reduced_data = data_i.iloc[:, :-1]
    selected_features = reduced_data.columns[model.get_support()].to_list()

################################################################################################################

    # SVM
    model = SVC()
    kernel = ['poly', 'rbf', 'sigmoid']
    C = [50, 10, 1.0, 0.1, 0.01]
    gamma = [1, 0.1, 0.01, 0.001]
    grid = dict(kernel=kernel, C=C, gamma=gamma)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train)
    # summarize result
    print(grid_result.best_params_)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, config in zip(means, params):
        config = str(config)
        if config in svm_parameters:
            svm_parameters[config].append(mean)
        else:
            svm_parameters[config] = [mean]


################################################################################################################
    model = MLPClassifier()
    solver=['lbfgs', 'sgd', 'adam']
    hidden_layer_sizes=[(5,), (10,), (20,),  (30,)]
    learning_rate_init = [0.0001, 0.001, 0.01,]
    max_iter=[30, 100, 200]
    grid = dict(solver=solver, hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init,
              max_iter=max_iter)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train)
    print(grid_result.best_params_)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, config in zip(means, params):
        config = str(config)
        if config in mlp_paramters:
            mlp_paramters[config].append(mean)
        else:
            mlp_paramters[config] = [mean]

################################################################################################################

    # RandomForestClassifier
    model = RandomForestClassifier()
    n_estimators = [10, 100, 1000]
    max_features = ['sqrt', 'log2']
    # define grid search
    grid = dict(n_estimators=n_estimators, max_features=max_features)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train)
    print(grid_result.best_params_)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, config in zip(means, params):
        config = str(config)
        if config in rf_paramters:
            rf_paramters[config].append(mean)
        else:
            rf_paramters[config] = [mean]

################################################################################################################

    # GradientBoostingClassifier
    model = GradientBoostingClassifier()
    n_estimators = [10, 100, 1000]
    learning_rate = [0.001, 0.01, 0.1]
    subsample = [0.5, 0.7, 1.0]
    max_depth = [3, 7, 9]
    # define grid search
    grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, config in zip(means, params):
        config = str(config)
        if config in xg_paramters:
            xg_paramters[config].append(mean)
        else:
            xg_paramters[config] = [mean]

################################################################################################################

    # BaggingClassifier
    model = BaggingClassifier()
    max_samples = [0.05, 0.1, 0.2, 0.5]
    n_estimators = [10, 100, 1000]
    # define grid search
    grid = dict(n_estimators=n_estimators, max_samples=max_samples)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train)
    # summarize results
   # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print(grid_result.best_params_)
  #  path = os.path.join(store_parameters, f"{i}.txt")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, config in zip(means, params):
        config = str(config)
        if config in bagg_paramters:
            bagg_paramters[config].append(mean)
        else:
            bagg_paramters[config] = [mean]

################################################################################################################

for k in svm_parameters.keys():
    svm_parameters[k] = np.array(svm_parameters[k]).mean()

for k in mlp_paramters.keys():
    mlp_paramters[k] = np.array(mlp_paramters[k]).mean()

for k in rf_paramters.keys():
    rf_paramters[k] = np.array(rf_paramters[k]).mean()

for k in xg_paramters.keys():
    xg_paramters[k] = np.array(xg_paramters[k]).mean()

for k in bagg_paramters.keys():
    bagg_paramters[k] = np.array(bagg_paramters[k]).mean()

fo = open(SVM_OUT_PATH, "w")
for k, v in svm_parameters.items():
    fo.write(str(k) + ' >>> '+ str(v) + '\n\n')

fo = open(MLP_OUT_PATH, "w")
for k, v in mlp_paramters.items():
    fo.write(str(k) + ' >>> '+ str(v) + '\n\n')

fo = open(RF_OUT_PATH, "w")
for k, v in rf_paramters.items():
    fo.write(str(k) + ' >>> '+ str(v) + '\n\n')

fo = open(XG_OUT_PATH, "w")
for k, v in xg_paramters.items():
    fo.write(str(k) + ' >>> '+ str(v) + '\n\n')

fo = open(BAGG_OUT_PATH, "w")
for k, v in bagg_paramters.items():
    fo.write(str(k) + ' >>> ' + str(v) + '\n\n')




