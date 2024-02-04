import opensmile
import pandas as pd
import os
#conds activate opensmile

input_dir = '/export/b01/afavaro/IS_2024/TAUKADIAL-24/training/train_audios_16k_no_diarization/'
out_dir = '/export/b01/afavaro/IS_2024/TAUKADIAL-24/training/feats/compare/no_diarization/'

all_files = [os.path.join(input_dir, elem) for elem in os.listdir(input_dir)]
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

dfs = []  # List to store DataFrames extracted from each audio file

for file in all_files:
    y = smile.process_file(file)
    dfs.append(pd.DataFrame(y))

# Concatenate all DataFrames into a single DataFrame
concatenated_df = pd.concat(dfs, ignore_index=True)
out_file = os.path.join(out_dir, 'compare_features.csv')
concatenated_df.to_csv(out_file)