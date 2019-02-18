import numpy as np
import os
lists = os.listdir('mirflickr25k_annotations_v080')
labels = np.zeros([25000,24], dtype=np.bool_)
count = 0
for type in lists:
    image_list = open('mirflickr25k_annotations_v080/'+type, 'r', encoding='UTF-8').readlines()
    for image_name in image_list:
        labels[int(image_name)-1, count] = 1
    count += 1
np.save('labels.npy', labels)
