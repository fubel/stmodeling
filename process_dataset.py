# This code hase been acquired from TRN-pytorch repository
# 'https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py'
# which is prepared by Bolei Zhou
#
# Processing the raw dataset of Jester
#
# generate the meta files:
#   category.txt:               the list of categories.
#   train_videofolder.txt:      each row contains [videoname num_frames classIDX]
#   val_videofolder.txt:        same as above
#
# Created by Bolei Zhou, Dec.2 2017

import os
import pdb

ROOT_DATASET_JESTER = '/usr/home/sut/datasets/jester/'
ROOT_DATASET = '/usr/home/sut/datasets/jester/rgb'

ROOT_DATASET_STH = '/usr/home/sut/datasets/something-something-v2/'
ROOT_DATASET_STH_RGB = '/usr/home/sut/datasets/something-something-v2/extracted-frames'

dataset_name = 'something-something-v2'
with open('%s%s-labels.csv' % (ROOT_DATASET_STH, dataset_name)) as f:
    lines = f.readlines()
categories = []
for line in lines:
    line = line.rstrip()
    categories.append(line)
categories = sorted(categories)
with open(os.path.join(ROOT_DATASET_STH,'category.txt'), 'w') as f:
    f.write('\n'.join(categories))

dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[category] = i

files_input = ['%s%s-validation.csv' % (ROOT_DATASET_STH, dataset_name), '%s%s-train.csv' % (ROOT_DATASET_STH,dataset_name)]
files_output = ['val_videofolder.txt', 'train_videofolder.txt']
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(filename_input) as f:
        lines = f.readlines()
    folders = []
    idx_categories = []
    for line in lines:
        line = line.rstrip()
        items = line.split(';')
        folders.append(items[0])
        idx_categories.append(os.path.join(str(dict_categories[items[1]])))
    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        curIDX = idx_categories[i]
        # counting the number of frames in each video folders
        dir_files = os.listdir(os.path.join(ROOT_DATASET_STH_RGB, curFolder))
        output.append('%s %d %d' % (curFolder, len(dir_files), int(curIDX)))
        print('%d/%d' % (i, len(folders)))
    with open(os.path.join(ROOT_DATASET_STH,filename_output), 'w') as f:
        f.write('\n'.join(output))
