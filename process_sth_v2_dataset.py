# This code hase been acquired from TRN-pytorch repository
# 'https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py'
# which is prepared by Bolei Zhou
#
# Processing the raw dataset of Something Something V2
#
# generate the meta files:
#   category.txt:               the list of categories.
#   train_videofolder.txt:      each row contains [videoname num_frames classIDX]
#   val_videofolder.txt:        same as above
#
# Created by Bolei Zhou, Dec.2 2017

import os
import pdb
import json

ROOT_DATASET = '/usr/home/sut/datasets/something-something-v2/'
ROOT_DATASET_FRAMES = '/usr/home/sut/datasets/something-something-v2/extracted-frames'

dataset_name = 'something-something-v2'
with open(f'{ROOT_DATASET}{dataset_name}-labels.json') as labels_json:
    dict_categories = json.load(labels_json)
with open(os.path.join(ROOT_DATASET,'category.txt'), 'w') as f:
    f.write('\n'.join(dict_categories.keys()))

files_input = ['%s%s-validation.json' % (ROOT_DATASET, dataset_name), '%s%s-train.json' % (ROOT_DATASET,dataset_name)]
files_output = ['val_videofolder.txt', 'train_videofolder.txt']
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(filename_input) as f:
        lines = json.load(filename_input)
    folders = []
    idx_categories = []
    for line in lines:
        folders.append(line['id'])
        label = str(line['template']).replace('[','')
        label = label.replace(']','')
        idx_categories.append(os.path.join(str(dict_categories[label])))
    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        curIDX = idx_categories[i]
        # counting the number of frames in each video folders
        dir_files = os.listdir(os.path.join(ROOT_DATASET_FRAMES, curFolder))
        output.append('%s %d %d' % (curFolder, len(dir_files), int(curIDX)))
        print('%d/%d' % (i, len(folders)))
    with open(os.path.join(ROOT_DATASET,filename_output), 'w') as f:
        f.write('\n'.join(output))
