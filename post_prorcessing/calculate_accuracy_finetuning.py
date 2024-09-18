import numpy as np
import os
import pandas as pd
import sys

if __name__ == "__main__":

    input_dir = sys.argv[1]
    all_res = [os.path.join(input_dir, elem) for elem in os.listdir(input_dir)]

    all_accs = []
    for data in all_res:
        all_accs.append(pd.read_csv(data)['accuracy'].tolist()[0])

    print(all_accs)
    print('ACCURACY:')
    print(round(np.sum(all_accs)/len(all_accs), 2))