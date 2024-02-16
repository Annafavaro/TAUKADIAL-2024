
sp_subset = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/training_speaker_division_helin/all.json'
tr_path_all = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/data_tianyu/transcripts_prompts_refined/all/'
output = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/all/'

import json
import numpy as np
import os
import pandas as pd

def create_data_frame(list_vals):
    names = []
    transcripts  = []
    labels = []
    mmse = []
    for elem in list_vals:
       # print(elem)
        names.append(elem[0])
        transcripts.append(elem[1])
        labels.append(elem[2])
        mmse.append(elem[3])
    dict = {'idx': names,'label': labels,  'sentences': transcripts}
    df = pd.DataFrame(dict)

    return df

n_folds_names = []
n_folds_data = []
all_folds_info = []

read_dict = json.load(open(sp_subset))
for key, values in read_dict.items():
    #  print(key, values)
    fold_info_general = []
    fold = list((read_dict[key]).keys())
    n_folds_names.append(list([os.path.basename(sp) for sp in fold]))
    fold_info = read_dict[key]  # get data for
    for sp in fold_info:
        fold_info_general.append(
            [sp, os.path.join(tr_path_all, sp.split('.wav')[0] + '.txt'), (fold_info[sp])['label'],
             (fold_info[sp])['mmse']])
    all_folds_info.append(fold_info_general)


print(n_folds_names[0])
folds = []
for fold in all_folds_info:
    data_fold = []  # %
    for speaker in fold:
        name = speaker[0]
        label_row = speaker[2]
        mmse = speaker[3]
        feat = open((speaker[1])).read().strip()
        # print(label_row, row['path_feat'])
        feat = [name, feat, label_row, mmse]
        data_fold.append(feat)
    folds.append(data_fold)


# For fold 1
data_train_1 = folds[:8][0]
data_val_1 = folds[8:9][0]
data_test_1 = folds[9:][0]

# For fold 2
data_train_2 = (folds[1:-1])[0]
data_val_2 = folds[-1:][0]
data_test_2 = folds[:1][0]

# For fold 3
data_train_3 = folds[2:][0]
data_val_3 = folds[:1][0]
data_test_3 = folds[1:2][0]

# For fold 4
data_train_4 = (folds[3:] + folds[:1])[0]
data_val_4 = folds[1:2][0]
data_test_4 = folds[2:3][0]

# For fold 5
data_train_5 = (folds[4:] + folds[:2])[0]
data_val_5 = folds[2:3][0]
data_test_5 = folds[3:4][0]

# For fold 6
data_train_6 =(folds[5:] + folds[:3])[0]
data_val_6 = folds[3:4][0]
data_test_6 = folds[4:5][0]

# For fold 7
data_train_7 = (folds[6:] + folds[:4])[0]
data_val_7 = folds[4:5][0]
data_test_7 = folds[5:6][0]

# For fold 8
data_train_8 = (folds[7:] + folds[:5])[0]
data_val_8 = folds[5:6][0]
data_test_8 = folds[6:7][0]

# For fold 9
data_train_9 = (folds[8:] + folds[:6])[0]
data_val_9 = folds[6:7][0]
data_test_9 = folds[7:8][0]

# For fold 10
data_train_10 = (folds[9:] + folds[:7])[0]
data_val_10 = folds[7:8][0]
data_test_10 = folds[8:9][0]

for i in range(1, 11):
    print(i)
    pd_train_X, pd_val, pd_test = create_data_frame(eval(f"data_train_{i}")), create_data_frame(
        eval(f"data_val_{i}")), create_data_frame(eval(f"data_test_{i}"))

    out_path_tr = os.path.join(output, f'cv_{i}', 'train.csv')
    out_path_dev = os.path.join(output, f'cv_{i}', 'dev.csv')
    out_path_test = os.path.join(output, f'cv_{i}', 'test.csv')

    pd_train_X.to_csv(out_path_tr)
    pd_val.to_csv(out_path_dev)
    pd_test.to_csv(out_path_test)
