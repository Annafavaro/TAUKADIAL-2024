out_path = '/export/b16/afavaro/Results_new_analysis/Multimodal_IS_2024/similarity_scores/'
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import os
import gensim.downloader as api
import pandas as pd

words = ['Jar', 'Curtain', 'Cupboard', 'Counter', 'Cloth', 'Window', 'Water',
         'Stool', 'Girl', 'Mother', 'Outside', 'Boy', 'Cookie']

model = api.load("word2vec-google-news-300")
combinations = list(itertools.combinations(words, 2))


# Include combinations of each word with itself
for word in words:
    combinations.append((word, word))

similarity_matrix = []
for word1, word2 in combinations:
    vec1 = model[word1]
    vec2 = model[word2]
    cos_val = cosine_similarity([vec1], word2)
    similarity_matrix.append([word1, word2, cos_val[0][0] ])
    print(f'similarity between {word1} and {word2} is ---> {cos_val}')


out_path_file = os.path.join(out_path, 'sim_scores.csv')
df_similarity = pd.DataFrame(similarity_matrix, columns=['Word 1', 'Word 2', 'Similarity'])
similarity_matrix_df = df_similarity.pivot(index='Word 1', columns='Word 2', values='Similarity')

similarity_matrix_df.to_csv(out_path_file)
print("Similarity matrix saved to similarity_matrix.csv")