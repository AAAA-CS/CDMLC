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
#定义一个以 HalfWidth 为半径的正方形窗口，用于从输入数据中提取样本。这个窗口的大小是 (2 * HalfWidth + 1) x (2 * HalfWidth + 1)。在这个函数中，HalfWidth 的值决定了从输入数据中提取的样本的大小和数量。
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, HalfWidth):
    print(Data_Band_Scaler.shape) # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape  #610 340 103

    '''label start'''
    num_class = int(np.max(GroundTruth)) # 九类
    data_band_scaler = utils.flip(Data_Band_Scaler) #（1830 1020 103）
    groundtruth = utils.flip(GroundTruth) #（1830 1020）
    del Data_Band_Scaler
    del GroundTruth

    # HalfWidth 从输入数据中提取特定区域的样本，以便后续的处理和分析。
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]     #（618 348）
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth:]    #（618 348 103）

    [Row, Column] = np.nonzero(G)   # 42776，42776
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)      # 42776
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {}        # Data Augmentation
    m = int(np.max(G))          # 九类
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS         #每类标记数 5
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)
    # 这个循环的作用是根据每个类别的样本索引，生成训练和测试样本的索引，并进行数据增强的处理。
    for i in range(m):   #循环遍历每一个类别
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1] #通过遍历 Row 和 Column 中的索引，找到属于当前类别 i + 1 的样本的索引，并存储在 indices 中。
        np.random.shuffle(indices)   #随机打乱 indices 中的索引顺序。
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]     #将打乱后的索引列表中的前 nb_val 个索引作为当前类别的训练样本索引，并存储在 train[i] 中。
        da_train[i] = []   #初始化一个空列表 da_train[i]，用于存储数据增强后的训练样本索引
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):  #循环计算数据增强的次数
            da_train[i] += indices[:nb_val]  #将当前类别的前 nb_val 个索引添加到数据增强训练样本索引列表 da_train[i] 中
        test[i] = indices[nb_val:]   #将剩余的索引作为当前类别的测试样本索引，并存储在 test[i] 中

    train_indices = []
    test_indices = []
    da_train_indices = []
    # 这个循环的作用是将每个类别的训练、测试和数据增强训练样本的索引汇总到一个大的索引列表中。
    for i in range(m):
        train_indices += train[i]  # 将当前类别的训练样本索引 train[i] 添加到总的训练样本索引列表 train_indices 中
        test_indices += test[i]   # 将当前类别的测试样本索引 test[i] 添加到总的测试样本索引列表 test_indices 中。
        da_train_indices += da_train[i]  # 将当前类别的数据增强训练样本索引 da_train[i] 添加到总的数据增强训练样本索引列表 da_train_indices 中。
    np.random.shuffle(test_indices)   # 随机打乱test_indices中索引顺序

    print('the number of train_indices:', len(train_indices))  # 9类 每类五个 共45
    print('the number of test_indices:', len(test_indices))  # 42731   一共42776个样本，45个当作训练样本， 42731个当作测试样本
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 将9类，每类五个训练样本，增强到每类200个训练样本
    print('labeled sample indices:', train_indices)  #将四十五个标记样本打印出来

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    #为每个训练和测试样本构建一个图像数据块，并将其存储到 imdb['data'] 中，并同时为每个样本分配相应的标签并存储到 imdb['Labels'] 中
    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.') #imdb data(9,9,103,42776) Labels 42776 set 全是1

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
