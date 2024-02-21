# conda activate lealla

# YES
device = 'cpu'

import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text
import numpy as np
import os
import sys
from numpy import save

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    encoder = hub.KerasLayer("https://www.kaggle.com/models/google/lealla/frameworks/TensorFlow2/variations/lealla-base/versions/1")
    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]

    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
    for sentences in all_sents:
        base_name = os.path.basename(sentences).split(".txt")[0]
        print(base_name)
        # sentences = open(sentences, 'r', encoding="utf-8",errors='ignore').read().strip().lower()
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip()#.lower()
            sentences = tf.constant([sentences])
            embeds = encoder(sentences)
            embeds = embeds.numpy()
            save(output_dir + base_name + '.npy', embeds)







