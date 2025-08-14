import torch
import torch.nn as nn
import torch.nn.functional as F
from model.loss_caco import CaContrast_loss

CaContrast = CaContrast_loss()

# query_labels [3,7,1,..]9类，每类19个共171个，query_features[171,128]
def loss_caco_cal(query_labels, query_features, PL_queue_train, queue_train, GPU, DIC_NUM_CLASSES, dict_len, label_rux, queue_label_mapping):
    loss_caco = CaContrast_cal (query_labels.detach ().clone (), query_features,
                                PL_queue_train.cuda (GPU).detach ().clone (), queue_train.cuda (GPU).detach ().clone (),
                                GPU, label_rux, queue_label_mapping)
    if [PL_queue_train != -1][0].sum() != DIC_NUM_CLASSES*dict_len:
        # print('dict is loading.....  ', int([PL_queue_train != -1][0].sum()), '/', DIC_NUM_CLASSES*dict_len)
        loss_caco = 0
    return loss_caco

# query_labels query_features PL_queue_train queue_train
def CaContrast_cal(label_label_aug1, feature, labels, feature_ma, GPU, label_rux, queue_label_mapping):
    with torch.no_grad():
        # [bsz, n_samples]
        label_label_aug1 = (label_label_aug1.clone().type(torch.FloatTensor)).unsqueeze(0)
        labels = (labels.clone().type(torch.FloatTensor)).view(labels.shape[0], -1)  # [1,17]
    # 19*128-》1*19*1*128
    feature = F.normalize (feature, dim=1)
    feature = feature.unsqueeze(0).unsqueeze(2)
    with torch.no_grad():
        # 队列特征1*128*17*1  1*128*17
        feature_ma = F.normalize(feature_ma.view(feature_ma.shape[0], feature_ma.shape[1], -1), dim=1)   #
        feature_ma = feature_ma.transpose(1, 2).unsqueeze(2)         # 1*128*17-》[1,17,1,128]
    # 查询特征，查询标签，训练队列，队列标签
    loss = CaContrast(features=feature, labels=label_label_aug1, features_2=feature_ma, labels_2=labels, GPU=GPU, label_rux=label_rux, queue_label_mapping=queue_label_mapping)
    return loss

def entropy_cal(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    return - torch.mul(v, torch.log2(v + 1e-30))


# queue: [1, 128, 27,50] PL_queue：[1, 27, 50] mom_support_features_src=torch.Size([9, 128]), support_labels_src=tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
def queue_update(queue, PL_queue, queue_label_mapping, f_src_main, src_labels, src_label_rux, f_tar_main, tar_labels, tar_label_rux, NUM_CLASSES):
    # 遍历所有类别标签
    for i in range(NUM_CLASSES):  # 27
        # 处理源域数据
        for j in range(torch.numel(src_labels)):   # 9
            if src_label_rux[int(src_labels[j])] == queue_label_mapping[i]:
                queue[0, :, i, 0] = f_src_main[j]
                PL_queue[0, i, 0] = i
                # 滚动队列，进行队列更新操作
                queue[0, :, i, :] = torch.roll(queue[0, :, i, :], -1, 1)
                PL_queue[0, i, :] = torch.roll(PL_queue[0, i, :], -1, 0)

        # 处理目标域数据
        for j in range(torch.numel(tar_labels)):
            if tar_label_rux[int(tar_labels[j])] == queue_label_mapping[i]:
                queue[0, :, i, 0] = f_tar_main[j]
                PL_queue[0, i, 0] = i
                queue[0, :, i, :] = torch.roll(queue[0, :, i, :], -1, 1)
                PL_queue[0, i, :] = torch.roll(PL_queue[0, i, :], -1, 0)
    queue_train = queue.view (queue.shape[0], queue.shape[1], -1)[:, :, PL_queue.squeeze (0).view (-1) != -1].unsqueeze (3)      # 1,128,17,1
    # print(PL_queue != -1)
    PL_queue_train = PL_queue[PL_queue != -1].unsqueeze (0).unsqueeze (2)   # [1,17,1]   1 250 1
    return queue, PL_queue, queue_train, PL_queue_train



