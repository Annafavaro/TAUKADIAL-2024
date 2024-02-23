import numpy as np
import os
import pandas as pd
#a = np.array([0,0,1,1,1])   # actual labels
#b = np.array([1,1,0,0,1])   # predicted labels
#
#correct = (a == b)
#accuracy = correct.sum() / correct.size

list_results = '/export/b01/afavaro/INTERSPEECH_2024/TAUKADIAL-24/training/finetuning/results/mono/chinese/'

all_res = [os.path.join(list_results, elem)  for elem in os.listdir(list_results)]

all_accs = []
for data in all_res:
    all_accs.append(pd.read_csv(data)['accuracy'].tolist()[0])

print(all_accs)
print(np.sum(all_accs)/len(all_accs))