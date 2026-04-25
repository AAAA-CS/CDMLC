import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CaContrast_loss(nn.Module):
    def __init__(self, temperature=0.5, contrast_mode='one', base_temperature=0.5):
        super(CaContrast_loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, features_2=None, labels_2=None, label_rux=None, queue_label_mapping=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_samples, n_views, ...].
            labels: ground truth of shape [bsz, n_samples].
            mask: contrastive mask of shape [bsz, n_samples, n_samples], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            features_2: historical features
            labels_2: corresponding labels
            reliability: logits_mask_score of shape [bsz, n_samples]
            cfg: configure file
        Returns:
            A loss scalar.
        """
        queue_label_mapping = dict(zip(queue_label_mapping.values(), queue_label_mapping.keys()))

        if len (features.shape) < 4:
            raise ValueError ('`features` needs to be [bsz, n_samples, n_views, ...],'
                              'at least 4 dimensions are required')
        if len (features_2.shape) < 4:
            raise ValueError ('`features` needs to be [bsz, n_samples, n_views, ...],'
                              'at least 4 dimensions are required')
        if len (features.shape) > 4:
            features = features.view (features.shape[0], features.shape[1], features.shape[2], -1)
        if len (features_2.shape) > 4:
            features_2 = features_2.view (features_2.shape[0], features_2.shape[1], features_2.shape[2], -1)

        n_samples = features.shape[1]
        if labels is not None and mask is not None:
            raise ValueError ('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # [bsz, bsz]
            mask = torch.eye (n_samples, dtype=torch.float32).to (device)
        elif labels is not None:
            labels = labels.contiguous ().view (labels.shape[0], -1, 1)
            labels_2 = labels_2.contiguous ().view (labels_2.shape[0], -1, 1)
            if labels.shape[1] != n_samples:
                raise ValueError ('Num of labels does not match num of features')
            if labels_2.shape[1] != features_2.shape[1]:
                raise ValueError ('Num of labels does not match num of features')
            for i in range (labels.shape[1]):
                labels[0][i][0] = float (queue_label_mapping[label_rux[int (labels[0][i][0])]])
            mask = torch.eq (labels, labels_2.transpose (1, 2)).float ().to (device)
        else:
            # [bsz, bsz]
            mask = mask.float ().to (device)

        contrast_count = features_2.shape[2]
        contrast_feature = torch.cat (torch.unbind (features_2, dim=2), dim=1)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, :, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError ('Unknown mode: {}'.format (self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div (
            torch.matmul (anchor_feature, contrast_feature.transpose (1, 2)),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max (anchor_dot_contrast, dim=2, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat (1, anchor_count, contrast_count)

        logits_mask = torch.ones_like (mask)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(2, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(2) / mask.shape[2]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view (features.shape[0], anchor_count, n_samples).mean ()

        return loss


CaContrast = CaContrast_loss()

# query_labels [3,7,1,..]9类，每类19个共171个，query_features[171,128]
def loss_caco_cal(query_labels, query_features, PL_queue_train, queue_train, label_rux, queue_label_mapping):
    loss_caco = CaContrast_cal (query_labels.detach ().clone (), query_features, PL_queue_train.detach ().clone (),
                                queue_train.detach ().clone (), label_rux, queue_label_mapping)

    return loss_caco

# query_labels query_features PL_queue_train queue_train
def CaContrast_cal(query_labels, query_features, labels, feature_ma, label_rux, queue_label_mapping):
    with torch.no_grad():
        # [bsz, n_samples]
        query_labels = (query_labels.clone().type(torch.FloatTensor)).unsqueeze(0)
        labels = (labels.clone().type(torch.FloatTensor)).view(labels.shape[0], -1)  # [1,17]

    feature = F.normalize (query_features, dim=1)
    feature = feature.unsqueeze(0).unsqueeze(2)
    with torch.no_grad():
        feature_ma = F.normalize(feature_ma.view(feature_ma.shape[0], feature_ma.shape[1], -1), dim=1)   #
        feature_ma = feature_ma.transpose(1, 2).unsqueeze(2)
    # 查询特征，查询标签，训练队列，队列标签
    loss = CaContrast(features=feature, labels=query_labels, features_2=feature_ma, labels_2=labels, label_rux=label_rux, queue_label_mapping=queue_label_mapping)
    return loss


def queue_update(queue, PL_queue, queue_label_mapping, f_src_main, src_labels, src_label_rux, f_tar_main, tar_labels, tar_label_rux, NUM_CLASSES):
    # 遍历所有类别标签
    for i in range(NUM_CLASSES):
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

    queue_train = queue.view (queue.shape[0], queue.shape[1], -1)[:, :, PL_queue.squeeze (0).view (-1) != -1].unsqueeze (3)
    PL_queue_train = PL_queue[PL_queue != -1].unsqueeze (0).unsqueeze (2)

    return queue, PL_queue, queue_train, PL_queue_train



