# conda activate mulitlingual_clip

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


def preprocess_text(input_text, lang='en'):
    # Load the appropriate spaCy model based on language
    if lang == 'en':
        nlp = spacy.load('en_core_web_sm')
    elif lang == 'zh':
        nlp = spacy.load('zh_core_web_sm')
    elif lang == 'es':
        nlp = spacy.load("es_core_news_sm")
    else:
        raise ValueError(f"Unsupported language: {lang}")
    # Tokenize and perform part-of-speech tagging
    doc = nlp(input_text)
    filtered_words = [token.text.lower() for token in doc if token.pos_.startswith('NOUN') and not token.is_stop]
    # Join the filtered words back into a string
    processed_text = ' '.join(filtered_words)
    return processed_text


def get_stats_data(transcripts_path, output_dir):

    all_sents = sorted([os.path.join(transcripts_path, elem) for elem in os.listdir(transcripts_path)])

    for sentences in all_sents:

        base_name = os.path.basename(sentences).split(".txt")[0]
        print(base_name)
        #sentences = open(sentences, 'r', encoding="utf-8",errors='ignore').read().strip().lower()
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip().lower()
            encoded_text = tokenizer.batch_encode_plus([str(sentences)], return_tensors="pt", truncation=True, max_length=512)
            text_embeddings = model_text(encoded_text)
            numpy_array = text_embeddings.numpy()
            save(output_dir + base_name + '.npy', numpy_array)


if __name__ == "__main__":

    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]

    model, preprocess = clip.load("ViT-L/14", device=device)
    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
    model_text = tf_multilingual_clip.MultiLingualCLIP.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    get_stats_data(input_dir, output_dir)














