# conda activate mulitlingual_clip

from sentence_transformers import SentenceTransformer
import sys
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn.functional as F
import torch
from numpy import save

#Mean Pooling - Take attention mask into account for correct averaging
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
            # Tokenize sentences
            encoded_input = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
            # Perform pooling. In this case, max pooling.
            # Perform pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            print(type(sentence_embeddings))
            print(sentence_embeddings.shape)
            #print(embeddings.shape)
            #print(output.keys())
            #embeddings = model.encode(sentences)
           # embeddings = embeddings.reshape(1, -1)
            #print(type(embeddings))
            #print(embeddings.shape)
            #save(output_dir + base_name + '.npy', embeddings)