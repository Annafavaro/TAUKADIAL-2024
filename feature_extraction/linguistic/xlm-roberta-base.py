# conda activate mulitlingual_clip
import sys
import os
from numpy import save
from transformers import AutoTokenizer, XLMRobertaModel
import torch
from transformers import AutoTokenizer, AutoModel

# YES:
# LANGUAGES: ENGLISH AND CHINESE, AMONG OTHERS
if __name__ == "__main__":

    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
    for sentences in all_sents:
        base_name = os.path.basename(sentences).split(".txt")[0]
        print(base_name)
        #sentences = open(sentences, 'r', encoding="utf-8",errors='ignore').read().strip().lower()
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip()#.lower()
            inputs = tokenizer(sentences, return_tensors="pt")
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state #take last hidden state
            sentence_embedding = torch.mean(last_hidden_states, dim=1) # mean pooling
            sentence_embedding = sentence_embedding.detach().numpy()  # dim: (1, 768)
            print(sentence_embedding.shape)
            save(output_dir + base_name + '.npy', sentence_embedding)