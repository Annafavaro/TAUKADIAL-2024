import pandas as pd
import os
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np
import pandas as pd
import random
import numpy as np
import random
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def selective_averaging(mmse_predictions, VA_iter, threshold):
    if VA_iter == 1:  # 1 is Controls
        mmse_predictions_filtered = [pred for pred in mmse_predictions if pred >= threshold]
        if not mmse_predictions_filtered:
            return np.max(mmse_predictions)  # No predictions for control group or above threshold
        else:
            return np.mean(mmse_predictions_filtered)

    elif VA_iter == 0:  # 0 is Alzheimer
        mmse_predictions_filtered = [pred for pred in mmse_predictions if pred < threshold]
        if not mmse_predictions_filtered:
            return np.min(mmse_predictions)  # No predictions for AD group or below threshold
        else:
            return np.mean(mmse_predictions_filtered)


def majority_voting(ad_predictions):
    ad_probs = np.mean(ad_predictions, axis=0)
    if np.max(ad_probs) >= 0.5:
        return 1  # control
    else:
        return 0  # AD


def predict(ad_models_preds, mmse_models_preds, threshold=26):
    iter = 0
    # prediction results of AD models
    A_iter = list(ad_models_preds)
    # print(A_iter)
    # prediction results of MMSE models
    MMSE = mmse_models_preds  # [model.predict() for model in mmse_models]
    VA_iter = majority_voting(A_iter)
    # print(MMSE, VA_iter)
    while True:
        iter += 1
        M_hat = selective_averaging(MMSE, VA_iter, threshold)
        NA = 0 if M_hat < threshold else 1
        A_iter.append(NA)
        # MMSE = MMSE_iter
        VA_iter_1 = majority_voting(A_iter)
        if VA_iter == VA_iter_1:
            break
        else:
            VA_iter = VA_iter_1

    return VA_iter, M_hat


final_preds = []
final_mmse = []

preds = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/interpretable/prediction/'
all_files_preds = [os.path.join(preds, elem) for elem in os.listdir(preds)]
mmse_folds = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/interpretable/regression/'
all_files_mmse = [os.path.join(mmse_folds, elem) for elem in os.listdir(mmse_folds)]

# feat1 = 'pause'
feat2 = 'ling'
feat3 = 'open'

gts_preds = []
gts_mmse = []

lists_values_preds = []
list_mmse = []
#
for element in sorted(all_files_preds):
    if feat1 in element or feat2 in element:
        print(element)
        lists_values_preds.append(pd.read_csv(element)['predictions'].tolist())
        gts_preds.append(pd.read_csv(element)['truth'].tolist())

print(" ")
for element in sorted(all_files_mmse):
    if feat1 in element or feat2 in element:
        print(element)
        list_mmse.append(pd.read_csv(element)['score'].tolist())
        gts_mmse.append(pd.read_csv(element)['truth'].tolist())

lists_values_mat = np.concatenate([lists_values_preds], axis=1)
list_mmse_mat = np.concatenate([list_mmse], axis=1)

transposed_array1 = np.transpose(lists_values_mat)
transposed_array2 = np.transpose(list_mmse_mat)
for i in zip(transposed_array1, transposed_array2):
    VA_iter, M_hat = predict(i[0], i[1])
    final_preds.append(VA_iter)
    final_mmse.append(M_hat)
    # print(M_hat)
