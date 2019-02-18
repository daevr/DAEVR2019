import json
from nltk.corpus import stopwords
import pickle
import numpy as np
stopword = set(stopwords.words('english'))
captions = json.load(open('captions_train2014.json', 'r'))
annotations = captions['annotations']
diciton = {}
for t in annotations:
    caption = t['caption'].lower()
    caption = caption.split(' ')
    for word in caption:
        if len(word)>0:
            if word not in diciton.keys():
                diciton[word] = 1
            else:
                diciton[word] += 1
items = diciton.items()
items = sorted(items, key= lambda d:d[1], reverse=True)
for i in range(len(items)):
    if items[i][0] in stopword:
        del diciton[items[i][0]]
items = diciton.items()
items = sorted(items, key= lambda d:d[1], reverse=True)
diciton.clear()
items = items[0:2000]
count = 0
for item in items:
    diciton[item[0]] = count
    count += 1
pickle.dump(diciton, open('diction.pkl', 'wb'))
bow = {}
for t in annotations:
    caption = t['caption'].lower()
    id = t['image_id']
    caption = caption.split(' ')
    for word in caption:
        if len(word)>0:
            if word in diciton.keys():
                if id in bow.keys():
                    bow[id][diciton[word]] = 1
                else:
                    bow[id] = np.zeros([2000], np.int8)
                    bow[id][diciton[word]] = 1
pickle.dump(bow, open('bow.pkl', 'wb'))
items = bow.items()
items = sorted(items, key= lambda d:d[0])
bow = [w[1] for w in items]
bow = np.array(bow)
np.save('bow.npy', bow)