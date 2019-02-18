import numpy as np
k_text = 3
k_image = 3
data_list = np.load('data_list.npy')
np.random.shuffle(data_list)
train_data_list = data_list[0:10000]
test_data_list = data_list[10000:12000]
np.save('train_data_list.npy', train_data_list)
np.save('test_data_list.npy', test_data_list)
bow = np.load('bow.npy')
labels = np.load('labels.npy')
vgg_features = np.load('vgg_19_mirflickr_features.npy')
vgg_features = np.tanh((vgg_features)/np.std(vgg_features))
dist = np.load('vgg_dist.npy')
dist_new = np.zeros([len(train_data_list),len(train_data_list)], np.float32)
for i in range(len(train_data_list)):
    for j in range(len(train_data_list)):
        dist_new[i,j] = dist[train_data_list[i],train_data_list[j]]
vgg_features_target = np.zeros_like(vgg_features, np.float32)
for i in range(len(train_data_list)):
    neb = np.argpartition(dist_new[i], k_image)[0:k_image]
    vgg_features_target[train_data_list[i]] = np.mean(vgg_features[train_data_list[neb]], 0)
np.save('vgg_19_mirflickr_features_target.npy', vgg_features_target)

dist = np.load('bow_dist.npy')
dist_new = np.zeros([len(train_data_list),len(train_data_list)], np.float32)
for i in range(len(train_data_list)):
    for j in range(len(train_data_list)):
        dist_new[i,j] = dist[train_data_list[i],train_data_list[j]]
bow_target = np.zeros_like(bow, np.float32)
for i in range(len(train_data_list)):
    neb = np.argpartition(dist_new[i], k_text)[0:k_text]
    bow_target[train_data_list[i]] = np.mean(bow[train_data_list[neb]], 0)
np.save('bow_target.npy', bow_target)