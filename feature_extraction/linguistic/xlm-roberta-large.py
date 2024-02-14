# conda activate mulitlingual_clip

from sentence_transformers import SentenceTransformer
import sys
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# YES:
# LANGUAGES: ENGLISH AND CHINESE, AMONG OTHERS
if __name__ == "__main__":

    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]
    model = SentenceTransformer('sentence-transformers/xlm-roberta-large')
    sentences = ["This is an example sentence"]
    embeddings = model.encode(sentences)
    print(embeddings)
    print(embeddings.shape)











   #tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
   #model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

   #all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
   #for sentences in all_sents:
   #    base_name = os.path.basename(sentences).split(".txt")[0]
   #    print(base_name)
   #    # sentences = open(sentences, 'r', encoding="utf-8",errors='ignore').read().strip().lower()
   #    with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
   #        sentences = file.read().strip()#.lower()
   #        # Tokenize sentences
   #        batch_dict = tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')
   #        outputs = model(**batch_dict)
   #        print(outputs.keys())
   #        embeddings = average_pool(outputs.logits, batch_dict['attention_mask'])

   #        ## normalize embeddings
   #        embeddings = F.normalize(embeddings, p=2, dim=1)
   #        embeddings = embeddings.detach().numpy()
   #        print(type(embeddings))
   #        print(embeddings.shape)

   #        #save(output_dir + base_name + '.npy', embeddings)