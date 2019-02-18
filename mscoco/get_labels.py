import json
import pickle
import numpy as np
instances = json.load(open('instances_train2014.json', 'r'))
category_map = {}
category = instances['categories']
annotation = instances['annotations']
del instances
count = 0
for cat in category:
    category_map[cat['id']] = count
    count += 1

label = {}
for t in annotation:
    cat_id = t['category_id']
    im_id = t['image_id']
    if im_id in label.keys():
        label[im_id][category_map[cat_id]] = 1
    else:
        label[im_id] = np.zeros([81], np.int8)
        label[im_id][category_map[cat_id]] = 1

pickle.dump(label, open('label.pkl', 'wb'))
items = label.items()
items = sorted(items, key= lambda d:d[0])
label = [w[1] for w in items]
label = np.array(label)
np.save('label.npy', label)