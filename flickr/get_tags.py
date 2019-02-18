import numpy as np
import os
import re
import pickle
lists = os.listdir('tags')
diction = {}
tag_list = [1] * 25000
for tag_doc in lists:
    text = open('tags/'+tag_doc, 'r', encoding='UTF-8').read()
    text = re.split('[ \n]',text)
    for word in text:
        word = re.sub('[ \n\t]', '', word)
        if len(word)<1:
            continue
        if word not in diction.keys():
            diction[word] = 1
        else:
            diction[word] += 1
key_list = list(diction.keys())
count = 0
for key in key_list:
    if diction[key]<20:
        del diction[key]
    else:
        diction[key] = count
        count += 1
print(len(diction))
for tag_doc in lists:
    text = open('tags/'+tag_doc, 'r', encoding='UTF-8').read()
    text = re.split('[\n]',text)
    temp = []
    for word in text:
        word = re.sub('[ \n\t]', '', word)
        if len(word)>0 and word in diction.keys():
            temp.append(word)
    tag_list[int(re.sub('[^0-9]', '',tag_doc))-1] = temp
data_list = []
for i in range(len(tag_list)):
    if(len(tag_list[i])>0):
        data_list.append(i)
data_list = np.array(data_list)
print(len(data_list))
np.save('data_list.npy', data_list)
pickle.dump(diction, open('diction.pkl', 'wb'))
pickle.dump(tag_list, open('tag_list.pkl', 'wb'))
