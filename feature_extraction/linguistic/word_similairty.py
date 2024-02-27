from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import gensim.downloader as api
from gensim.models import KeyedVectors

words = ['CookieJar', 'Curtain', 'Cupboard', 'Dishware', 'Dishcloth', 'Window', 'Water',
         'Stool', 'Girl', 'Mother', 'Outside', 'Boy', 'Cookie']

#model = api.load("word2vec-google-news-300")
#Word2Vec.load("word2vec.model")
#word_vectors = model.wv
#v_mango = model['mango']
#print(v_mango)
#print('compute similarity')

combinations = list(itertools.combinations(words, 2))

# Include combinations of each word with itself
for word in words:
    combinations.append((word, word))

# Store combinations as lists
pair_lists = [[pair[0], pair[1]] for pair in combinations]

# Print all combinations
for pair_list in pair_lists:
    print(pair_list)