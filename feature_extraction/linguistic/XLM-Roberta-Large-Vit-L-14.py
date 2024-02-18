# conda activate mulitlingual_clip

# YES
device = 'cpu'

import torch
import os
import numpy as np
import re
from numpy import save
print("Torch version:", torch.__version__)
import transformers
import clip
import spacy
from multilingual_clip import tf_multilingual_clip
import sys


def get_stats_data(transcripts_path, output_dir):

    all_sents = sorted([os.path.join(transcripts_path, elem) for elem in os.listdir(transcripts_path)])[107:]

    for sentences in all_sents:

        base_name = os.path.basename(sentences).split(".txt")[0]
        print(base_name)
        #sentences = open(sentences, 'r', encoding="utf-8",errors='ignore').read().strip().lower()
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip()#.lower()
            encoded_text = tokenizer.batch_encode_plus([str(sentences)], return_tensors="pt", truncation=True, max_length=512)
            text_embeddings = model_text(encoded_text)
            numpy_array = text_embeddings.numpy()
            print(numpy_array.shape)
            save(output_dir + base_name + '.npy', numpy_array)


if __name__ == "__main__":

    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]

    model, preprocess = clip.load("ViT-L/14", device=device)
    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
    model_text = tf_multilingual_clip.MultiLingualCLIP.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    get_stats_data(input_dir, output_dir)














