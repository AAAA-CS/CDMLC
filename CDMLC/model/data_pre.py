import numpy as np
import utils
import math
import torch
import imp
import argparse
import os
import torch.nn as nn

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument('--config', type=str, default=os.path.join('./config', 'paviaU.py'),
                    help='config file with parameters of the experiment. '
                            'It is assumed that the config file is placed under the directory ./config')
args = parser.parse_args()

config = imp.load_source("", args.config).config
train_opt = config['train_config']

TEST_LSAMPLE_NUM_PER_CLASS = train_opt['test_lsample_num_per_class']


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, patch_size):
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class, HalfWidth=patch_size//2)
    # train_datas, train_labels = train_loader.__iter__().next()
    train_datas, train_labels = iter(train_loader).__next__()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    return train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column, nTrain


# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, HalfWidth):
    print(Data_Band_Scaler.shape) # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape  #610 340 103

    '''label start'''
    num_class = int(np.max(GroundTruth)) 
    data_band_scaler = utils.flip(Data_Band_Scaler) #（1830 1020 103）
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]     #（618 348）
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth:]    #（618 348 103）

    [Row, Column] = np.nonzero(G)   # 42776，42776
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)     
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {}       
    m = int(np.max(G))        
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS         
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)
 
    for i in range(m):  
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1] 
        np.random.shuffle(indices)   
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]     
        da_train[i] = []  
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1): 
            da_train[i] += indices[:nb_val] 
        test[i] = indices[nb_val:]  
    train_indices = []
    test_indices = []
    da_train_indices = []
   
    for i in range(m):
        train_indices += train[i]  
        test_indices += test[i]  
        da_train_indices += da_train[i] 
    np.random.shuffle(test_indices)  

    print('the number of train_indices:', len(train_indices)) 
    print('the number of test_indices:', len(test_indices)) 
    print('the number of train_indices after data argumentation:', len(da_train_indices))  
    print('labeled sample indices:', train_indices)  

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.') 

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain

