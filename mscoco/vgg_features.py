#import vgg
import os
import tensorflow as tf
from PIL import Image
import numpy as np
#vgg = vgg.Vgg19()
#images = tf.placeholder(tf.float32, [None, 224, 224, 3])
#vgg.build(images)
#features = vgg.fc7
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
        #print(name)
        temp_im = np.array(Image.open('D:\\train2014\\' + name).resize((224, 224)))
        if len(temp_im.shape) == 3:
            temp.append(int(name[15:27]))
temp = np.array(temp)
np.save('image_list.npy', temp)

