from sklearn.decomposition import PCA
from gensim import corpora, models
import pickle
import numpy as np
tokens = pickle.load(open('tag_list.pkl', 'rb'))
dictionary = corpora.Dictionary(tokens)
texts = [dictionary.doc2bow(text) for text in tokens]
tfidf_model = models.TfidfModel(texts, normalize=False)
tfidf = np.zeros([len(tokens),1386], np.float32)
for i in range(len(tokens)):
    temp = tfidf_model[texts[i]]
    for topic in temp:
        tfidf[i, topic[0]] = topic[1]
np.save('tf_idf.npy', tfidf)