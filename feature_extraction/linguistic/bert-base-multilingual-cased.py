# conda activate mulitlingual_clip

from sentence_transformers import SentenceTransformer
import sys
import os
from transformers import BertTokenizer, BertModel
import re
from numpy import save
# yes
#no--> cannot be used
if __name__ == "__main__":

    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained("bert-base-multilingual-cased")

    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
    for sentences in all_sents:
        base_name = os.path.basename(sentences).split(".txt")[0]
        print(base_name)
        # sentences = open(sentences, 'r', encoding="utf-8",errors='ignore').read().strip().lower()
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip().lower()
            print(sentences)
            encoded_input = tokenizer(sentences, return_tensors='pt')
            output = model(**encoded_input)
            embeddings = output['last_hidden_state']
            print(type(embeddings))
            print(embeddings.shape)
           # embeddings = embeddings.numpy()
           # save(output_dir + base_name + '.npy', embeddings)
