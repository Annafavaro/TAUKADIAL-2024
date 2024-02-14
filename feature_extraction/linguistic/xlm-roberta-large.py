# conda activate mulitlingual_clip

from sentence_transformers import SentenceTransformer
import sys
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
import re
import torch
from numpy import save

# YES:
# LANGUAGES: ENGLISH AND CHINESE, AMONG OTHERS
if __name__ == "__main__":

    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
    for sentences in all_sents[140:]:
        base_name = os.path.basename(sentences).split(".txt")[0]
        print(base_name)
        # sentences = open(sentences, 'r', encoding="utf-8",errors='ignore').read().strip().lower()
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip()#.lower()
            encoded_input = tokenizer(sentences, return_tensors='pt')
            # forward pass
            output = model(**encoded_input)
            print(type(output))
            print(output.keys())
            #embeddings = model.encode(sentences)
           # embeddings = embeddings.reshape(1, -1)
            #print(type(embeddings))
            #print(embeddings.shape)
            #save(output_dir + base_name + '.npy', embeddings)