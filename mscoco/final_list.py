import numpy as np
import pickle
image_list = np.load('image_list.npy')
feature = np.load('vgg_19_mscoco_features.npy')
label = pickle.load(open('label.pkl', 'rb'))
bow = pickle.load(open('bow.pkl', 'rb'))
new_label = []
new_bow = []
new_feature = []
count = 0
for id in image_list:
    if id in label.keys() and id in bow.keys():
        new_feature.append(feature[count])
        new_label.append(label[id])
        new_bow.append(bow[id])
    count += 1
new_label = np.array(new_label)
new_bow = np.array(new_bow)
new_feature = np.array(new_feature)
np.save('label.npy', new_label)
np.save('bow.npy', new_bow)
np.save('feature.npy', new_feature)