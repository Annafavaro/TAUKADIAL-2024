# conda activate mulitlingual_clip

# no
from sentence_transformers import SentenceTransformer
import sys
import os
import re
from numpy import save


 all_sents at once
if __name__ == "__main__":

    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]

    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
    for sentences in all_sents:
        base_name = os.path.basename(sentences).split(".txt")[0]
        print(base_name)
        # sentences = open(sentences, 'r', encoding="utf-8",errors='ignore').read().strip().lower()
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip().lower()
            model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
            embeddings = model.encode(sentences)
            print(type(embeddings))
            #numpy_array = embeddings.numpy()
            save(output_dir + base_name + '.npy', embeddings)


#from sentence_transformers import SentenceTransformer
#import sys
#import os
#import re
#import numpy as np
#from numpy import save
#
#if __name__ == "__main__":
#
#    if len(sys.argv) != 3:
#        print("Usage: python script.py input_directory output_directory")
#        sys.exit(1)
#
#    input_dir = sys.argv[1]
#    output_dir = sys.argv[2]
#
#    if not os.path.isdir(input_dir):
#        print("Input directory does not exist.")
#        sys.exit(1)
#
#    model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
#    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
#
#    for sentences in all_sents:
#        print(sentences)
#        base_name = os.path.basename(sentences).split(".txt")[0]
#        sentences = open(sentences, 'r', encoding="utf-8").read().strip().lower()
#        if '.' in sentences:
#            sents = sentences.split('.')
#        elif '。' in sentences:
#            sents = sentences.split('。')
#
#        embeddings = np.zeros(shape=(len(sents), 768))
#        for i in range(len(sents)):
#            if sents[i] != '':
#                embeddings[i, :] = model.encode(sents[i])
#        hidden_layer_avg = np.mean(embeddings, axis=0)
#        hidden_layer_avg = hidden_layer_avg.reshape((1, 768))
#        out_path = os.path.join(output_dir, base_name + '.npy')
#        save(out_path, hidden_layer_avg)