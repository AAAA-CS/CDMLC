import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.distributions as D
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import torch_clustering
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import time
import utils
import importlib.util
import sys
from torch.utils.data import DataLoader, RandomSampler
from model import feature
from model import  category_level as CL
from model import category_distribution_level as CDL

parser = argparse.ArgumentParser (description="CDMLC")
parser.add_argument ('--config', type=str, default=os.path.join ('./config', 'Indian_pines.py'),
                     help='config file with parameters of the experiment. '
                          'It is assumed that the config file is placed under the directory ./config')
args = parser.parse_args ()

spec = importlib.util.spec_from_file_location("config_module", args.config)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

config = config_module.config
# 文件路径
data_path = config['data_path']
source_data = config['source_data']
target_data = config['target_data']
target_data_gt = config['target_data_gt']
TEST_LSAMPLE_NUM_PER_CLASS = config['test_lsample_num_per_class']

train_opt = config['train_config']
# 标签映射
src_label_mapping = train_opt['src_label_mapping']
tar_label_mapping = train_opt['tar_label_mapping']
queue_label_mapping_src = train_opt['queue_label_mapping_src']
queue_label_mapping_tar = train_opt['queue_label_mapping_tar']
# 特征维度
patch_size = train_opt['patch_size']
SRC_INPUT_DIMENSION = train_opt['src_input_dim']  
TAR_INPUT_DIMENSION = train_opt['tar_input_dim']  
N_DIMENSION = train_opt['n_dim']
emb_size = train_opt['d_emb']
# 元任务设置
EPISODE = train_opt['episode']
CLASS_NUM = train_opt['class_num']  
SHOT_NUM_PER_CLASS = train_opt['shot_num_per_class']  
QUERY_NUM_PER_CLASS = train_opt['query_num_per_class']
# 类别数量
TEST_CLASS_NUM = train_opt['test_class_num']
DIC_NUM_CLASSES_SRC = train_opt['dic_num_classes_src']
DIC_NUM_CLASSES_TAR = train_opt['dic_num_classes_tar']
n_clusters = train_opt['n_clusters']
# 训练参数
LEARNING_RATE = train_opt['lr']
DIC_LEN = train_opt['dic_len']
move_momentum = train_opt['move_momentum']  
SAMPLE_SIZE = train_opt['sample_size']

Lambda = train_opt['lambda']
Beta = train_opt['Beta']

GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load source data
with open (os.path.join (data_path, source_data), 'rb') as handle:
    source_imdb = pickle.load (handle)
print (source_imdb.keys ()) 
print (source_imdb['Labels']) 

# process source data
data_train = source_imdb['data']  
labels_train = source_imdb['Labels']  
print (data_train.shape)  
print (labels_train.shape) 
keys_all_train = sorted (list (set (labels_train)))
label_encoder_train = {}
for i in range (len (keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i

train_set = {}
for class_, path in zip (labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append (path)
print (train_set.keys ())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print ("Num classes for source domain datasets: " + str (len (data))) 
print (data.keys ())
data = utils.sanity_check (data)   # 去掉少于200个样本的类
print ("Num classes of the number of class larger than 200: " + str (len (data)))
print (data.keys ())

for class_ in data:
    for i in range (len (data[class_])):
        image_transpose = np.transpose (data[class_][i], (2, 0, 1))
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data  
print (len (metatrain_data.keys ()), metatrain_data.keys ())
del data

# source domain adaptation data
print (source_imdb['data'].shape)  # (77592, 9, 9, 128)
source_imdb['data'] = source_imdb['data'].transpose ((1, 2, 3, 0))  # (9, 9, 128, 77592)
print (source_imdb['data'].shape)
print (source_imdb['Labels'])

# target data
# load target data
test_data = os.path.join (data_path, target_data)
test_label = os.path.join (data_path, target_data_gt)
Data_Band_Scaler, GroundTruth = utils.load_data (test_data, test_label)

# run 10 times
nDataSet = 1
acc = np.zeros ([nDataSet, 1])
A = np.zeros ([nDataSet, CLASS_NUM])
k = np.zeros ([nDataSet, 1])
seeds = [1233, 1335, 1336, 1337, 1338]

for iDataSet in range(nDataSet):
    print('iDataSet:', iDataSet)
    np.random.seed (seeds[iDataSet])
    print('seed:', seeds[iDataSet])

    last_accuracy = 0.0
    best_episode = 0

    # load target domain data for training and testing
    train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column, nTrain = utils.get_target_dataset (
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=TEST_CLASS_NUM,
        shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS, patch_size=patch_size)

    # 特征提取器
    feature_encoder = feature.Network(patch_size, emb_size)
    feature_encoder_optim = torch.optim.Adam (feature_encoder.parameters (), lr=LEARNING_RATE)
    feature_encoder.apply (feature.weights_init)
    feature_encoder.to (GPU)
    feature_encoder.train ()
    # 动量编码器
    momentum_encoder = feature.Network (patch_size, emb_size)
    momentum_encoder.load_state_dict (feature_encoder.state_dict ())
    momentum_encoder.to (GPU)
    momentum_encoder.train()
    # 损失函数
    crossEntropy = nn.CrossEntropyLoss().to(GPU)
    # 源域队列
    queue_src = torch.zeros (1, emb_size, DIC_NUM_CLASSES_SRC, DIC_LEN)
    PL_queue_src = torch.ones (1, DIC_NUM_CLASSES_SRC, DIC_LEN, dtype=torch.int64) * (-1)  
    # 目标域队列
    queue_tar = torch.zeros (1, emb_size, DIC_NUM_CLASSES_TAR, DIC_LEN)
    PL_queue_tar = torch.ones (1, DIC_NUM_CLASSES_TAR, DIC_LEN, dtype=torch.int64) * (-1)  

    total_hit_src, total_num_src, total_hit_tar, total_num_tar = 0.0, 0.0, 0.0, 0.05

    print("Training...")
    train_start = time.time ()
    for episode in range (EPISODE):
        for param_q, param_k in zip (feature_encoder.parameters (), momentum_encoder.parameters ()):
            param_k.data = param_k.data.clone () * move_momentum + param_q.data.clone () * (1. - move_momentum)
        for buffer_q, buffer_k in zip (feature_encoder.buffers (), momentum_encoder.buffers ()):
            buffer_k.data = buffer_q.data.clone ()

        # get few-shot classification samples  class_num = 9、shot_num_per_class = 1、query_num_per_class = 19
        task = utils.Task (metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS, src_label_mapping)
        support_dataloader_src = utils.get_HBKC_data_loader (task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
        query_dataloader_src = utils.get_HBKC_data_loader (task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)
        src_label_rux = task.label_rux

        task = utils.Task (target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS, tar_label_mapping)
        support_dataloader_tar = utils.get_HBKC_data_loader (task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
        query_dataloader_tar = utils.get_HBKC_data_loader (task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)
        tar_label_rux = task.label_rux

        # sample datas
        supports_src, support_labels_src = iter(support_dataloader_src).__next__()
        querys_src, query_labels_src = iter(query_dataloader_src).__next__()

        supports_tar, support_labels_tar = iter(support_dataloader_tar).__next__()
        querys_tar, query_labels_tar = iter(query_dataloader_tar).__next__()

        sampled_test_datas = torch.zeros(SAMPLE_SIZE, TAR_INPUT_DIMENSION, 9, 9)
        total_samples = len(test_loader.dataset)
        indices = random.sample(range(total_samples), SAMPLE_SIZE)
        sampled_list = []
        for i in indices:
            data_point = test_loader.dataset[i][0]  # 获取数据 (NumPy array)
            if isinstance(data_point, np.ndarray):
                data_point = torch.from_numpy(data_point).float()  # 转换为 Tensor
            sampled_list.append(data_point)

        sampled_test_datas = torch.stack(sampled_list, dim=0).to(GPU)

        # calculate features
        support_features_src = feature_encoder (supports_src.to (GPU))
        query_features_src = feature_encoder (querys_src.to (GPU))
        support_features_tar = feature_encoder (supports_tar.to (GPU), domain='target')
        query_features_tar = feature_encoder (querys_tar.to (GPU), domain='target')
        gmm_features = feature_encoder (sampled_test_datas.to (GPU), domain='target')

        with torch.no_grad ():
            mom_support_features_src = momentum_encoder (supports_src.to (GPU))
            mom_support_features_tar = momentum_encoder (supports_tar.to (GPU), domain='target')
          
            queue_src, PL_queue_src, queue_train_src, PL_queue_train_src = CL.queue_update (queue_src, PL_queue_src,
                                                                                              queue_label_mapping_src,
                                                                                              mom_support_features_src,
                                                                                              support_labels_src,
                                                                                              src_label_rux,
                                                                                              mom_support_features_tar,
                                                                                              support_labels_tar,
                                                                                              tar_label_rux,
                                                                                              DIC_NUM_CLASSES_SRC)
            queue_tar, PL_queue_tar, queue_train_tar, PL_queue_train_tar = CL.queue_update (queue_tar, PL_queue_tar,
                                                                                              queue_label_mapping_tar,
                                                                                              mom_support_features_src,
                                                                                              support_labels_src,
                                                                                              src_label_rux,
                                                                                              mom_support_features_tar,
                                                                                              support_labels_tar,
                                                                                              tar_label_rux,
                                                                                              DIC_NUM_CLASSES_TAR)

        # calculate prototype
        if SHOT_NUM_PER_CLASS > 1:
            support_proto_src = [support_features_src[i].reshape (CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean (dim=1) for i in range (len (support_features_src))]
            support_proto_tar = [support_features_tar[i].reshape (CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean (dim=1) for i in range (len (support_features_tar))]
        else:
            support_proto_src = support_features_src
            support_proto_tar = support_features_tar

        '''few-shot learning'''
        logits_src = utils.euclidean_metric (query_features_src, support_proto_src)
        f_loss_src = crossEntropy (logits_src, query_labels_src.long ().to (GPU))
        logits_tar = utils.euclidean_metric (query_features_tar, support_proto_tar)
        f_loss_tar = crossEntropy (logits_tar, query_labels_tar.long ().to (GPU))
        f_loss = f_loss_src + f_loss_tar

        ''' category-level'''
        loss_caco_src = CL.loss_caco_cal (query_labels_src, query_features_src, PL_queue_train_src.to (GPU), queue_train_src.to (GPU),
                                            src_label_rux, queue_label_mapping_src)
        loss_caco_tar = CL.loss_caco_cal (query_labels_tar, query_features_tar, PL_queue_train_tar.to (GPU), queue_train_tar.to (GPU),
                                            tar_label_rux, queue_label_mapping_tar)
        loss_caco = loss_caco_src + loss_caco_tar

        ''' category-distribution-level'''
        means_src = []
        covariances_src = []
        for label in range(n_clusters):
            label_indices = query_labels_src == label
            label_features = query_features_src[label_indices]
            mean = torch.mean(label_features, dim=0)
            covariances = torch.var(label_features, dim=0)
            means_src.append(mean)
            covariances_src.append(covariances)
        means_src = torch.stack(means_src)
        covariances_src = torch.stack(covariances_src)

        gmm_tar = torch_clustering.PyTorchGaussianMixture(n_clusters=n_clusters)
        means_tar, covariances_tar, weights_tar = gmm_tar.fit_predict(gmm_features)

        values_src = [src_label_rux[key] for key in range (n_clusters)]
        values_tar = tar_label_mapping.values()
        positive_samples_src, negative_samples_src = CDL.match_samples(means_src, means_tar, values_src, values_tar)
        loss_mean_src = CDL.simclr_loss(means_src, positive_samples_src, negative_samples_src)

        positive_samples_tar, negative_samples_tar = CDL.match_samples(means_tar, means_src, values_tar, values_src)
        loss_mean_tar = CDL.simclr_loss(means_tar, positive_samples_tar, negative_samples_tar)

        loss_mean = loss_mean_src + loss_mean_tar

        cov_loss1 = torch.mean(torch.norm(covariances_src - covariances_tar, dim=-1))
        cov_loss2 = torch.abs(torch.mean(covariances_src, dim=-1)) + torch.abs(torch.mean(covariances_tar, dim=-1))
        cov_loss2 = torch.mean(cov_loss2)
        cov_loss = cov_loss1 + cov_loss2

        loss_gmm = loss_mean + cov_loss

        '''total loss'''
        loss = f_loss + Lambda*loss_caco + Beta*loss_gmm

        # Update parameters
        feature_encoder.zero_grad ()
        loss.backward ()
        feature_encoder_optim.step ()

        total_hit_src += torch.sum (torch.argmax (logits_src, dim=1).cpu () == query_labels_src).item ()
        total_num_src += querys_src.shape[0]
        total_hit_tar += torch.sum (torch.argmax (logits_tar, dim=1).cpu () == query_labels_tar).item ()
        total_num_tar += querys_tar.shape[0]

        if (episode + 1) % 100 == 0:
            print (
                'episode {:>3d}:  f_loss: {:6.4f}, loss_caco: {:6.4f}, loss_gmm: {:6.4f}, acc_src {:6.4f},' 
                ' acc_tar {:6.4f}, total_loss: {:6.4f}'.format (
                    episode + 1,
                    f_loss.item (),
                    loss_caco.item (),
                    loss_gmm.item(),
                    total_hit_src / total_num_src,
                    total_hit_tar / total_num_tar,
                    loss.item ()))

        if (episode + 1) % 1000 == 0 or episode == 0:
            # test
            print ("Testing ...")
            train_end = time.time ()
            feature_encoder.eval ()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array ([], dtype=np.int64)
            labels = np.array ([], dtype=np.int64)

            train_datas, train_labels = iter(train_loader).__next__()
            train_features = feature_encoder (train_datas.to (GPU), domain='target')

            max_value = train_features.max ()
            min_value = train_features.min ()
            print (max_value.item ())
            print (min_value.item ())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier (n_neighbors=1)
            KNN_classifier.fit (train_features.cpu ().detach ().numpy (), train_labels)
            test_labels_all, feature_emb = [], []
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                test_features = feature_encoder (test_datas.to (GPU), domain='target')
                feature_emb.append (test_features.cpu ().detach ().numpy ())
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict (test_features.cpu ().detach ().numpy ())

                test_labels = test_labels.numpy ()
                test_labels_all.append (test_labels)
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range (batch_size)]

                total_rewards += np.sum (rewards)
                counter += batch_size

                predict = np.append (predict, predict_labels)
                labels = np.append (labels, test_labels)

                accuracy = total_rewards / 1.0 / counter
                accuracies.append (accuracy)

            test_accuracy = 100. * total_rewards / len (test_loader.dataset)

            print ('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format (total_rewards, len (test_loader.dataset), 100. * total_rewards / len (test_loader.dataset)))
            test_end = time.time ()

            # Training mode
            feature_encoder.train ()
            if test_accuracy > last_accuracy:
                last_accuracy = test_accuracy
                best_episode = episode

                acc[iDataSet] = 100. * total_rewards / len (test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix (labels, predict)
                A[iDataSet, :] = np.diag (C) / np.sum (C, 1, dtype=float)
                k[iDataSet] = metrics.cohen_kappa_score (labels, predict)

            print ('best episode:[{}], best accuracy={}'.format (best_episode + 1, last_accuracy))

    print ('iter:{} best episode:[{}], best accuracy={}'.format (iDataSet, best_episode + 1, last_accuracy))
    print ("train time per DataSet(s): " + "{:.5f}".format (train_end - train_start))
    print ("accuracy list: ", acc)
    print ('***********************************************************************************')

AA = np.mean (A, 1)

AAMean = np.mean (AA, 0)
AAStd = np.std (AA)

AMean = np.mean (A, 0)
AStd = np.std (A, 0)

OAMean = np.mean (acc)
OAStd = np.std (acc)

kMean = np.mean (k)
kStd = np.std (k)
print ("train time per DataSet(s): " + "{:.5f}".format (train_end - train_start))
print ("test time per DataSet(s): " + "{:.5f}".format (test_end - train_end))
print ("accuracy list: ", acc)
print ("average OA: " + "{:.2f}".format (OAMean) + " +- " + "{:.2f}".format (OAStd))
print ("average AA: " + "{:.2f}".format (100 * AAMean) + " +- " + "{:.2f}".format (100 * AAStd))
print ("average kappa: " + "{:.4f}".format (100 * kMean) + " +- " + "{:.4f}".format (100 * kStd))
print ("accuracy for each class: ")
for i in range (CLASS_NUM):
    print ("Class " + str (i) + ": " + "{:.2f}".format (100 * AMean[i]) + " +- " + "{:.2f}".format (100 * AStd[i]))

best_iDataset = 0
for i in range (len (acc)):
    print ('{}:{}'.format (i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print ('best acc all={}'.format (acc[best_iDataset]))