import os
import torch
import torchvision
import torchvision.datasets as datasets

ROOT_JESTER = '/usr/home/sut/datasets/jester'
ROOT_SOMETHINGV2 = '/usr/home/sut/datasets/something-something-v2'

def return_jester(modality):
    filename_categories = os.path.join(ROOT_JESTER,'category.txt')
    filename_imglist_train = os.path.join(ROOT_JESTER,'train_videofolder.txt')
    filename_imglist_val = os.path.join(ROOT_JESTER,'val_videofolder.txt')
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = ROOT_JESTER
    elif modality == 'RGBFlow':
        prefix = '{:05d}.jpg'
        root_data = ROOT_JESTER
    else:
        print('no such modality:' + modality)
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = os.path.join(ROOT_SOMETHINGV2,'category.txt')
    root_data = ROOT_SOMETHINGV2
    if modality == 'RGB':
        filename_imglist_train = os.path.join(ROOT_SOMETHINGV2,'train_videofolder.txt')
        filename_imglist_val = os.path.join(ROOT_SOMETHINGV2,'val_videofolder.txt')
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        filename_imglist_train = os.path.join(ROOT_SOMETHINGV2,'train_videofolder.txt')
        filename_imglist_val = os.path.join(ROOT_SOMETHINGV2,'val_videofolder.txt')
        prefix = '{:06d}.jpg'
    else:
        print('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'something': return_somethingv2, 'jester': return_jester}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset ' + dataset)
    '''
    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    file_categories = os.path.join(ROOT_DATASET, file_categories)
    '''
    with open(file_categories) as f:
        lines = f.readlines()
    
    categories = [item.rstrip() for item in lines]
    return categories, file_imglist_train, file_imglist_val, root_data, prefix
