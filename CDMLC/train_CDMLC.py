import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.distributions as D
import torch.nn.functional as F
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
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
import time
import utils
import imp
from torch.utils.data import DataLoader, RandomSampler
from model import data_pre
from model import feature
from model import caco

# Hyper Parameters
parser = argparse.ArgumentParser (description="Few Shot Visual Recognition")
parser.add_argument ('--config', type=str, default=os.path.join ('./config', 'paviaU.py'),
                     help='config file with parameters of the experiment. '
                          'It is assumed that the config file is placed under the directory ./config')
args = parser.parse_args ()

# Hyper Parameters
config = imp.load_source ("", args.config).config
train_opt = config['train_config']
data_path = config['data_path']
save_path = config['save_path']
source_data = config['source_data']
target_data = config['target_data']
target_data_gt = config['target_data_gt']
src_label_mapping = config['src_label_mapping']
tar_label_mapping = config['tar_label_mapping']
queue_label_mapping_src = config['queue_label_mapping_src']
queue_label_mapping_tar = config['queue_label_mapping_tar']

patch_size = train_opt['patch_size'] 
emb_size = train_opt['d_emb']
SRC_INPUT_DIMENSION = train_opt['src_input_dim']  
TAR_INPUT_DIMENSION = train_opt['tar_input_dim']  
N_DIMENSION = train_opt['n_dim']
CLASS_NUM = train_opt['class_num']  
SHOT_NUM_PER_CLASS = train_opt['shot_num_per_class']  
QUERY_NUM_PER_CLASS = train_opt['query_num_per_class'] 
EPISODE = train_opt['episode']  
LEARNING_RATE = train_opt['lr']

GPU = config['gpu']
TEST_CLASS_NUM = train_opt['test_class_num'] 
TEST_LSAMPLE_NUM_PER_CLASS = train_opt['test_lsample_num_per_class']  

DIC_NUM_CLASSES_SRC = train_opt['dic_num_classes_src']
DIC_NUM_CLASSES_TAR = train_opt['dic_num_classes_tar']
DIC_LEN = train_opt['dic_len']
move_momentum = train_opt['move_momentum']  
SAMPLE_SIZE = train_opt['sample_size']
n_clusters = train_opt['n_clusters']
utils.same_seeds (0)

def _init_():
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()


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
print (keys_all_train)  
label_encoder_train = {}
for i in range (len (keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print (
    label_encoder_train)  

train_set = {}
for class_, path in zip (labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append (path)
print (train_set.keys ())  # dict_keys([5, 0, 8, 13, 7, 6, 4, 17, 10, 1, 3, 12, 16, 14, 2, 9, 11, 15, 18])
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print ("Num classes for source domain datasets: " + str (len (data))) 
print (data.keys ())  # dict_keys([5, 0, 8, 13, 7, 6, 4, 17, 10, 1, 3, 12, 16, 14, 2, 9, 11, 15, 18])
data = utils.sanity_check (data) 
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
print (source_imdb['Labels'])  # 0-18

# target data
# load target data
test_data = os.path.join (data_path, target_data)
test_label = os.path.join (data_path, target_data_gt)
Data_Band_Scaler, GroundTruth = utils.load_data (test_data, test_label)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find ('Conv') != -1:
        nn.init.xavier_uniform_ (m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_ ()
    elif classname.find ('BatchNorm') != -1:
        nn.init.normal_ (m.weight, 1.0, 0.02)
        m.bias.data.zero_ ()
    elif classname.find ('Linear') != -1:

        nn.init.xavier_normal_ (m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones (m.bias.data.size ())


crossEntropy = nn.CrossEntropyLoss ().to (GPU)
domain_criterion = nn.BCEWithLogitsLoss ().to (GPU)

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze (1).expand (n, m, -1)
    b = b.unsqueeze (0).expand (n, m, -1)
    logits = -((a - b) ** 2).sum (dim=2)
    return logits


def find_closest_indices(A, B):
    used_indices = []
    dist_matrix = torch.cdist(A, B, p=2)
    for idx in used_indices:
        dist_matrix[:, idx] = float('inf')
    closest_indices = torch.argmin(dist_matrix, dim=1)
    used_indices.extend(closest_indices.tolist())
    return closest_indices, used_indices

def match_samples(A, B, labels_A, labels_D):
    mask_D = torch.tensor([label in labels_D for label in labels_A], dtype=torch.bool)
    positive_samples = torch.zeros_like(A)
    positive_indices = torch.zeros(len(A), dtype=torch.long)
    closest_indices_D, used_indices = find_closest_indices(A[mask_D], B)
    positive_samples[mask_D] = B[closest_indices_D]
    device = positive_indices.device 
    closest_indices_D = closest_indices_D.to(device)
    positive_indices[mask_D] = closest_indices_D

    remaining_indices_A = torch.arange(len(A))[~mask_D]
    remaining_indices_B = torch.arange(len(B))
    remaining_indices_B = remaining_indices_B[~torch.isin(remaining_indices_B, torch.tensor(used_indices))]
    if len(remaining_indices_A) <= len(remaining_indices_B):
        selected_indices = remaining_indices_B[:len(remaining_indices_A)]
    else:
        raise ValueError("Not enough remaining elements in B to match remaining elements in A")

    positive_samples[remaining_indices_A] = B[selected_indices]
    positive_indices[remaining_indices_A] = selected_indices
    negative_samples = []
    for idx in range(len(A)):
        negative_indices = torch.cat((torch.arange(0, idx), torch.arange(idx + 1, len(A))))
        negative_samples.append(A[negative_indices])

    negative_samples = torch.stack(negative_samples)

    return positive_samples, negative_samples

def simclr_loss(anchors, positives, negatives, temperature=0.6):
    batch_size = anchors.size(0)
    embedding_size = anchors.size(1)
    num_negatives = negatives.size(1)

    positive_dist = torch.norm(anchors - positives, dim=-1) / temperature
    positive_sim = -positive_dist

    anchors_expanded = anchors.view(batch_size, 1, embedding_size).expand(-1, num_negatives, -1)
    negative_dist = torch.norm(anchors_expanded - negatives, dim=-1) / temperature
    negative_sim = -negative_dist

    logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long).to(anchors.device)  # 使用 anchors 的设备
    loss = F.cross_entropy(logits, labels)

    return loss

# run 10 times
nDataSet = 1
acc = np.zeros ([nDataSet, 1])  # acc[[0.]]
time_list = np.zeros ([nDataSet, 1])
A = np.zeros ([nDataSet, CLASS_NUM])
k = np.zeros ([nDataSet, 1])
best_predict_all = []
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None
class_map = np.zeros ((nDataSet,), dtype=object)
feature_emb_cell = np.zeros ((nDataSet,), dtype=object)
test_labels_cell = np.zeros ((nDataSet,), dtype=object)
test_labels_end, feature_emb_end = [], []

seeds = [1336, 1224, 1233, 1226, 1237,  1236, 1337, 1235, 1229, 1220]
for iDataSet in range(nDataSet):
    print('iDataSet:', iDataSet)
   
    np.random.seed (seeds[iDataSet])

    # load target domain data for training and testing
    train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column, nTrain = data_pre.get_target_dataset (
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=TEST_CLASS_NUM,
        shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS, patch_size=patch_size)

    feature_encoder = feature.Network (patch_size, emb_size)
    feature_encoder_optim = torch.optim.Adam (feature_encoder.parameters (), lr=LEARNING_RATE)  # weight_decay=train_opt['weight_decay'])
    feature_encoder.apply (weights_init)
    feature_encoder.to (GPU)
    feature_encoder.train ()

    momentum_encoder = feature.Network (patch_size, emb_size)
    momentum_encoder.load_state_dict (feature_encoder.state_dict ())  
    momentum_encoder.train ()
    momentum_encoder.to (GPU)

    queue_src = torch.zeros (1, emb_size, DIC_NUM_CLASSES_SRC, DIC_LEN)
    PL_queue_src = torch.ones (1, DIC_NUM_CLASSES_SRC, DIC_LEN, dtype=torch.int64) * (-1)  

    queue_tar = torch.zeros (1, emb_size, DIC_NUM_CLASSES_TAR, DIC_LEN)
    PL_queue_tar = torch.ones (1, DIC_NUM_CLASSES_TAR, DIC_LEN, dtype=torch.int64) * (-1)  

    print ("Training...")

    last_accuracy = 0.0
    last_accuracy_gnn = 0.0
    best_episdoe = 0
    train_loss = []
    total_hit_src, total_num_src, total_hit_tar, total_num_tar = 0.0, 0.0, 0.0, 0.05

    gmm_test_loader = test_loader
  
    sampled_test_datas = torch.zeros(SAMPLE_SIZE, TAR_INPUT_DIMENSION, 9, 9)
    # sampled_test_labels = []
    sample_size = SAMPLE_SIZE

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

        total_samples = len(gmm_test_loader.dataset)
       
        indices = random.sample(range(total_samples), sample_size)
 
        for i, (batch_datas, batch_labels) in enumerate(gmm_test_loader):
            for idx in indices:
                if idx < len(batch_datas):
                    sampled_test_datas[i] = batch_datas[idx]
            if len(sampled_test_datas) >= sample_size:
                break

        # calculate features
        support_features_src = feature_encoder (supports_src.to (GPU))
        query_features_src = feature_encoder (querys_src.to (GPU))
        support_features_tar = feature_encoder (supports_tar.to (GPU), domain='target')
        query_features_tar = feature_encoder (querys_tar.to (GPU), domain='target')

        gmm_features = feature_encoder (sampled_test_datas.to (GPU), domain='target')


        with torch.no_grad ():
            mom_support_features_src = momentum_encoder (supports_src.to (GPU))
            mom_support_features_tar = momentum_encoder (supports_tar.to (GPU), domain='target')
          
            queue_src, PL_queue_src, queue_train_src, PL_queue_train_src = caco.queue_update (queue_src, PL_queue_src,
                                                                                              queue_label_mapping_src,
                                                                                              mom_support_features_src[1],
                                                                                              support_labels_src,
                                                                                              src_label_rux,
                                                                                              mom_support_features_tar[1],
                                                                                              support_labels_tar,
                                                                                              tar_label_rux,
                                                                                              DIC_NUM_CLASSES_SRC)
            queue_tar, PL_queue_tar, queue_train_tar, PL_queue_train_tar = caco.queue_update (queue_tar, PL_queue_tar,
                                                                                              queue_label_mapping_tar,
                                                                                              mom_support_features_src[1],
                                                                                              support_labels_src,
                                                                                              src_label_rux,
                                                                                              mom_support_features_tar[1],
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
        logits_src = euclidean_metric (query_features_src[1], support_proto_src[1])
        f_loss_src = crossEntropy (logits_src, query_labels_src.long ().to (GPU))
        logits_tar = euclidean_metric (query_features_tar[1], support_proto_tar[1])
        f_loss_tar = crossEntropy (logits_tar, query_labels_tar.long ().to (GPU))
        f_loss = f_loss_src + f_loss_tar

        '''contrast learning'''
        loss_caco_src = caco.loss_caco_cal (query_labels_src, query_features_src[1], PL_queue_train_src, queue_train_src,
                                            GPU, DIC_NUM_CLASSES_SRC, DIC_LEN, src_label_rux, queue_label_mapping_src)
        loss_caco_tar = caco.loss_caco_cal (query_labels_tar, query_features_tar[1], PL_queue_train_tar, queue_train_tar,
                                            GPU, DIC_NUM_CLASSES_TAR, DIC_LEN, tar_label_rux, queue_label_mapping_tar)
        loss_caco = loss_caco_src + loss_caco_tar

        query_features_src = query_features_src[1]
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
        means_tar, covariances_tar, weights_tar = gmm_tar.fit_predict(gmm_features[1])

        values_src = [src_label_rux[key] for key in range (n_clusters)]
        values_tar = tar_label_mapping.values()
        positive_samples_src, negative_samples_src = match_samples(means_src, means_tar, values_src, values_tar)
        loss_mean_src = simclr_loss(means_src, positive_samples_src, negative_samples_src) * 0.1

        negative_samples_tar = []
        for idx in range(len(positive_samples_src)):
            negative_indices = torch.cat((torch.arange(0, idx), torch.arange(idx + 1, len(positive_samples_src))))
            negative_samples_tar.append(positive_samples_src[negative_indices])

        negative_samples_tar = torch.stack(negative_samples_tar)
        loss_mean_tar = simclr_loss( positive_samples_src, means_src, negative_samples_tar) * 0.1
        loss_mean = loss_mean_src + loss_mean_tar

        cov_loss1 = torch.mean(torch.norm(covariances_src - covariances_tar, dim=-1))
        cov_loss2 = torch.abs(torch.mean(covariances_src, dim=-1)) + torch.abs(torch.mean(covariances_tar, dim=-1))
        cov_loss2 = torch.mean(cov_loss2)
        cov_loss = cov_loss1 + cov_loss2

        loss_gmm = (loss_mean + cov_loss)*0.1

        '''total loss'''
        loss = f_loss + 0.5*loss_caco + 20*loss_gmm
        # Update parameters
        feature_encoder.zero_grad ()
        loss.backward ()
        feature_encoder_optim.step ()

        total_hit_src += torch.sum (torch.argmax (logits_src, dim=1).cpu () == query_labels_src).item ()
        total_num_src += querys_src.shape[0]
        total_hit_tar += torch.sum (torch.argmax (logits_tar, dim=1).cpu () == query_labels_tar).item ()
        total_num_tar += querys_tar.shape[0]

        if (episode + 1) % 100 == 0:
            train_loss.append (loss.item ())
            if loss_caco == 0:
                print (
                    'episode {:>3d}:  fsl loss: {:6.4f}, caco loss: {:6.4f}, loss_gmm: {:6.4f}, acc_src {:6.4f},'
                    ' acc_tar {:6.4f}, loss: {:6.4f}'.format (
                        episode + 1,
                        f_loss.item (),
                        0,
                        loss_gmm.item(),
                        total_hit_src / total_num_src,
                        total_hit_tar / total_num_tar,
                        loss.item ()))
            else:
                print (
                    'episode {:>3d}:  fsl loss: {:6.4f}, caco loss: {:6.4f}, loss_gmm: {:6.4f}, acc_src {:6.4f},' 
                    ' acc_tar {:6.4f}, loss: {:6.4f}'.format (
                        episode + 1,
                        f_loss.item (),
                        loss_caco.item (),
                        loss_gmm.item(),
                        total_hit_src / total_num_src,
                        total_hit_tar / total_num_tar,
                        loss.item ()))
                print ('fea_lr： {:6.6f}'.format (feature_encoder_optim.param_groups[0]['lr']))

        if (episode + 1) % 500 == 0 or episode == 0:
            # test
            print ("Testing ...")
            train_end = time.time ()
            feature_encoder.eval ()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array ([], dtype=np.int64)
            predict_gnn = np.array ([], dtype=np.int64)
            labels = np.array ([], dtype=np.int64)

            train_datas, train_labels = iter(train_loader).__next__()
            _, train_features = feature_encoder (Variable (train_datas).to (GPU), domain='target')

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

                _, test_features = feature_encoder (Variable (test_datas).to (GPU), domain='target')
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

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append (accuracy)

            test_accuracy = 100. * total_rewards / len (test_loader.dataset)

            print ('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format (total_rewards, len (test_loader.dataset), 100. * total_rewards / len (test_loader.dataset)))
            print ('seeds:', seeds[iDataSet])
            test_end = time.time ()

            # Training mode
            feature_encoder.train ()
            if test_accuracy > last_accuracy:
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len (test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix (labels, predict)
                A[iDataSet, :] = np.diag (C) / np.sum (C, 1, dtype=float)
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
                k[iDataSet] = metrics.cohen_kappa_score (labels, predict)

                feature_emb_end = np.concatenate (feature_emb)
                test_labels_end = np.concatenate (test_labels_all)

            print ('best episode:[{}], best accuracy={}'.format (best_episdoe + 1, last_accuracy))

    print ('iter:{} best episode:[{}], best accuracy={}'.format (iDataSet, best_episdoe + 1, last_accuracy))
    print ('iter:{} best episode:[{}], best accuracy_gnn={}'.format (iDataSet, best_episdoe + 1, last_accuracy_gnn))
    print ("train time per DataSet(s): " + "{:.5f}".format (train_end - train_start))
    print ("accuracy list: ", acc)
    print ('***********************************************************************************')
    for i in range (len (best_predict_all)):
        best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1
    class_map[iDataSet] = best_G[4:-4, 4:-4]
    time_list[iDataSet] = train_end - train_start
    feature_emb_cell[iDataSet] = feature_emb_end
    test_labels_cell[iDataSet] = test_labels_end

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

#################classification map################################
# #
# # for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
# #     best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1
# #
# # hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
# # for i in range(best_G.shape[0]):
# #     for j in range(best_G.shape[1]):
# #         if best_G[i][j] == 0:
# #             hsi_pic[i, j, :] = [0, 0, 0]
# #         if best_G[i][j] == 1:
# #             hsi_pic[i, j, :] = [0, 0, 1]
# #         if best_G[i][j] == 2:
# #             hsi_pic[i, j, :] = [0, 1, 0]
# #         if best_G[i][j] == 3:
# #             hsi_pic[i, j, :] = [0, 1, 1]
# #         if best_G[i][j] == 4:
# #             hsi_pic[i, j, :] = [1, 0, 0]
# #         if best_G[i][j] == 5:
# #             hsi_pic[i, j, :] = [1, 0, 1]
# #         if best_G[i][j] == 6:
# #             hsi_pic[i, j, :] = [1, 1, 0]
# #         if best_G[i][j] == 7:
# #             hsi_pic[i, j, :] = [0.5, 0.5, 1]
# #         if best_G[i][j] == 8:
# #             hsi_pic[i, j, :] = [0.65, 0.35, 1]
# #         if best_G[i][j] == 9:
# #             hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
# #         if best_G[i][j] == 10:
# #             hsi_pic[i, j, :] = [0.75, 1, 0.5]
# #         if best_G[i][j] == 11:
# #             hsi_pic[i, j, :] = [0.5, 1, 0.65]
# #         if best_G[i][j] == 12:
# #             hsi_pic[i, j, :] = [0.65, 0.65, 0]
# #         if best_G[i][j] == 13:
# #             hsi_pic[i, j, :] = [0.75, 1, 0.65]
# #         if best_G[i][j] == 14:
# #             hsi_pic[i, j, :] = [0, 0, 0.5]
# #         if best_G[i][j] == 15:
# #             hsi_pic[i, j, :] = [0, 1, 0.75]
# #         if best_G[i][j] == 16:
# #             hsi_pic[i, j, :] = [0.5, 0.75, 1]
#
# utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/IP_{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))