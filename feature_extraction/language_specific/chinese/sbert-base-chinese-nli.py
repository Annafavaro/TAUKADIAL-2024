# conda activate mulitlingual_clip

from sentence_transformers import SentenceTransformer
import sys
import os
import re
from numpy import save
# yes

device = 'cpu'

#no--> cannot be used
if __name__ == "__main__":

    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]
    model = SentenceTransformer('uer/sbert-base-chinese-nli').to('cpu')

    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
    for sentences in all_sents:
        base_name = os.path.basename(sentences).split(".txt")[0]
        print(base_name)
        # sentences = open(sentences, 'r', encoding="utf-8",errors='ignore').read().strip().lower()
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip().lower()
            embeddings = model.encode(sentences)
            embeddings = embeddings.reshape(1, -1)
            print(type(embeddings))
            print(embeddings.shape)
           # numpy_array = embeddings.numpy()
            save(output_dir + base_name + '.npy', embeddings)
