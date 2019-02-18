import numpy as np
k_text = 3
k_image = 3
labels = np.load('label.npy')
vgg_features = np.load('feature.npy')
vgg_features = np.tanh((vgg_features)/np.std(vgg_features))
bow = np.load('bow.npy')
data_list = np.arange(len(bow))
np.random.shuffle(data_list)
train_data_list = data_list[0:10000]
test_data_list = data_list[10000:15000]
np.save('train_data_list.npy', train_data_list)
np.save('test_data_list.npy', test_data_list)
train_vgg = vgg_features[train_data_list]
train_bow = bow[train_data_list]

norm = np.linalg.norm(train_vgg, axis=1)
dist_new = np.zeros([len(train_data_list),len(train_data_list)], np.float32)
for i in range(len(train_data_list)):
    dist_new[i] = -np.sum(train_vgg[i] * train_vgg, 1) / norm / \
              norm[i]
vgg_features_target = np.zeros_like(vgg_features, np.float32)
for i in range(len(train_data_list)):
    neb = np.argpartition(dist_new[i], k_image)[0:k_image]
    vgg_features_target[train_data_list[i]] = np.mean(vgg_features[train_data_list[neb]], 0)
np.save('vgg_features_target.npy', vgg_features_target)

norm = np.linalg.norm(train_bow, axis=1)
dist_new = np.zeros([len(train_data_list),len(train_data_list)], np.float32)
for i in range(len(train_data_list)):
    dist_new[i] = -np.sum(train_bow[i] * train_bow, 1) / norm / \
              norm[i]
bow_target = np.zeros_like(bow, np.float32)
for i in range(len(train_data_list)):
    neb = np.argpartition(dist_new[i], k_text)[0:k_text]
    bow_target[train_data_list[i]] = np.mean(bow[train_data_list[neb]], 0)
np.save('bow_target.npy', bow_target)