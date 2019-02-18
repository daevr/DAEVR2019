import vgg
import os
import tensorflow as tf
from PIL import Image
import numpy as np
vgg = vgg.Vgg19()
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg.build(images)
features = vgg.fc7
image_lists = os.listdir('D:\\train2014')
start = 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
all_features = []
temp = []
image_lists = sorted(image_lists)
with tf.Session(config=config) as session:
    for i in range(len(image_lists)):
        name = image_lists[i]
        print(name)
        temp_im = np.array(Image.open('D:\\train2014\\' + name).resize((224, 224)))
        if len(temp_im.shape) == 3:
            temp.append(temp_im)
        if len(temp)>=16 or i==len(image_lists)-1:
            temp = np.array(temp)
            temp = temp / 256
            _features = session.run(features, feed_dict={images: temp})
            all_features.append(_features)
            temp = []
all_features = np.concatenate(all_features, axis=0)
print(len(all_features))
np.save('vgg_19_mscoco_features.npy', all_features)

