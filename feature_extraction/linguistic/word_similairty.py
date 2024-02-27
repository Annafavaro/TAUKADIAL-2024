from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import gensim.downloader as api
from gensim.models import KeyedVectors

words = ['CookieJar', 'Curtain', 'Cupboard', 'Dishware', 'Dishcloth', 'Window', 'Water',
         'Stool', 'Girl', 'Mother', 'Outside', 'Boy', 'Cookie']

combinations = list(itertools.combinations(words, 2))
pair_lists = [[pair[0], pair[1]] for pair in combinations]
model = api.load("word2vec-google-news-300")
#Word2Vec.load("word2vec.model")


print('compute similarity')

for pair_list in pair_lists:
    print(pair_list)