import model
import numpy as np
import pickle
import tensorflow as tf
mAP_n = 10000
P_n = 1000
def HammingDistance(a, b):

    #c = np.logical_xor((np.sign(a)+1).astype(np.bool_),(np.sign(b)+1).astype(np.bool_))
    #dis = np.sum(c.astype(np.int32))
    dis = np.sum(np.square(a-b))
    return dis
def check(x,y):
    flag = bool(np.sum(np.logical_and(x, y)))
    return flag
def countMAP(result, train_label, po):
    AP = 0
    total_relevant = 0
    buffer_yes = np.zeros(mAP_n)
    Ns = np.arange(1, mAP_n+1, 1)
    for i in range(mAP_n):
        if check(train_label[result[i]], po):
            buffer_yes[i] = 1
            total_relevant += 1

    P = np.cumsum(buffer_yes)/Ns
    if sum(buffer_yes)!=0:
        AP += sum(P*buffer_yes)/sum(buffer_yes)
    return AP
def countP(result, train_label, po):
    P = 0
    for i in range(P_n):
        if check(train_label[result[i]], po):
            P += 1
    return P/P_n
def eval_total(binary_codes_train, binary_codes_test, train_label, test_label):
    lens1 = len(binary_codes_test)
    lens2 = len(binary_codes_train)
    P = 0
    mAP = 0
    dist = np.zeros((lens1, lens2), dtype=np.float32)
    binary_codes_train_norm = np.linalg.norm(binary_codes_train, axis=1)
    binary_codes_test_norm = np.linalg.norm(binary_codes_test, axis=1)
    for i in range(lens1):
        dist[i] = -np.sum(binary_codes_test[i] * binary_codes_train, 1) / binary_codes_train_norm / \
                  binary_codes_test_norm[i]
    results = []
    for i in range(lens1):
        results.append(np.argsort(dist[i]))
    for i in range(lens1):
        mAP += countMAP(results[i], train_label, test_label[i])
        P += countP(results[i], train_label, test_label[i])
    return  P/lens1, mAP/lens1

BATCH_SIZE = 128
tf_idf = np.load('bow.npy')
labels = np.load('label.npy')
vgg_features = np.load('feature.npy')
vgg_features = np.tanh((vgg_features)/np.std(vgg_features))
images = tf.placeholder(tf.float32, [None, 4096])
texts = tf.placeholder(tf.float32, [None, 2000])
model = model.luo(BATCH_SIZE,64)
text_embedding = model.txt_encoder(texts)
image_embedding = model.image_encoder(images)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
train_data_list = np.load('train_data_list.npy')
test_data_list = np.load('test_data_list.npy')
total_train_embedding = []
total_test_embedding = []
with tf.Session(config=config) as session:
    saver = tf.train.Saver()
    saver.restore(session, './referee_ae1.50_0.05')
    for i in range(int(len(train_data_list)/1000)):
        train_images = vgg_features[train_data_list[i*1000:min((i+1)*1000, len(train_data_list))]]
        _image_embedding = session.run(image_embedding, feed_dict={images:train_images})
        total_train_embedding.append(_image_embedding)
    total_train_embedding = np.concatenate(total_train_embedding, axis=0)
    for i in range(int(len(test_data_list)/1000)):
        test_texts = tf_idf[test_data_list[i*1000:min((i+1)*1000, len(test_data_list))]]
        _text_embedding = session.run(text_embedding, feed_dict={texts:test_texts})
        total_test_embedding.append(_text_embedding)
    total_test_embedding = np.concatenate(total_test_embedding, axis=0)

P, mAP = eval_total(total_train_embedding,total_test_embedding, labels[train_data_list], labels[test_data_list])
print('mAP')
print(mAP)
print('P')
print(P)
