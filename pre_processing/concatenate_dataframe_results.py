import os
import pandas as pd

###################### PREDICTION INTERPRETABLE ##########################

preds_eng = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language/english/interpretable/prediction/'
all_pred_eng = [os.path.join(preds_eng, elem) for elem in os.listdir(preds_eng)]
preds_china = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language/chinese/interpretable/prediction/'
all_pred_china = [os.path.join(preds_china, elem) for elem in os.listdir(preds_china)]
all_preds_int = all_pred_eng + all_pred_china

###################### PREDICTION NON-INTERPRETABLE ##########################

preds_nonint_chin = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language/chinese/non_interpretable/prediction/'
all_preds_nonint_chin = [os.path.join(preds_nonint_chin, elem) for elem in os.listdir(preds_nonint_chin)]
preds_nonint_eng = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language/english/non_interpretable/prediction/'
all_preds_nonint_eng = [os.path.join(preds_nonint_eng, elem) for elem in os.listdir(preds_nonint_eng)]
all_preds_non_int = all_preds_nonint_chin + all_preds_nonint_eng

###################### REGRESSION INTERPRETABLE ##########################

mmse_eng_int =  '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language/english/interpretable/regression/'
mmse_eng_int_all = [os.path.join(mmse_eng_int, elem) for elem in os.listdir(mmse_eng_int)]
mmse_chin_int =  '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language/chinese/interpretable/regression/'
mmse_chin_int_all = [os.path.join(mmse_chin_int, elem) for elem in os.listdir(mmse_chin_int)]
all_mmse_int = mmse_eng_int_all + mmse_chin_int_all

###################### REGRESSION NON-INTERPRETABLE ##########################

mmse_eng_nonint =  '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language/english/non_interpretable/regression/'
mmse_eng_nonint_all = [os.path.join(mmse_eng_nonint, elem) for elem in os.listdir(mmse_eng_nonint)]
mmse_chin_nonint = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language/chinese/non_interpretable/regression/'
mmse_chin_nonint_all = [os.path.join(mmse_chin_nonint, elem) for elem in os.listdir(mmse_chin_nonint)]
all_mmse_nonint = mmse_eng_nonint_all + mmse_chin_nonint_all

out_int_preds = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language_concatenated/interpretable/prediction/'
out_non_preds = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language_concatenated/non_interpretable/prediction/'
out_int_regr = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language_concatenated/interpretable/regression/'
out_non_regr = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/saved_predictions/results_per_language_concatenated/non_interpretable/regression/'

out_dir = out_non_regr

unique_preds = list(set(sorted([os.path.basename(elem) for elem in all_mmse_nonint])))
for unique in unique_preds:
    print(unique)
    chin =  pd.read_csv(os.path.join(mmse_chin_nonint, unique)).reset_index(drop=True)
    eng =   pd.read_csv(os.path.join(mmse_eng_nonint, unique)).reset_index(drop=True)
    tot = pd.concat([chin, eng],  ignore_index=True)
    out_path_file = os.path.join(out_dir, unique)
    tot.to_csv(out_path_file)