# conda activate mulitlingual_clip

from sentence_transformers import SentenceTransformer
import sys
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
import re
import torch
from numpy import save

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# YES:
# LANGUAGES: ENGLISH AND CHINESE, AMONG OTHERS
if __name__ == "__main__":

    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
    for sentences in all_sents:
        base_name = os.path.basename(sentences).split(".txt")[0]
        print(base_name)
        # sentences = open(sentences, 'r', encoding="utf-8",errors='ignore').read().strip().lower()
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip()#.lower()
            encoded_input = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
            # forward pass
            with torch.no_grad():
                output = model(**encoded_input)

            embeddings = mean_pooling(output, encoded_input['attention_mask'])
            emebddings = embeddings.numpy()
            print(type(embeddings))
            print(embeddings)
            print(embeddings.shape)
            #print(output.keys())
            #embeddings = model.encode(sentences)
           # embeddings = embeddings.reshape(1, -1)
            #print(type(embeddings))
            #print(embeddings.shape)
            #save(output_dir + base_name + '.npy', embeddings)