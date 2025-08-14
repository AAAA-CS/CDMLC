import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class CaContrast_loss(nn.Module):
    def __init__(self, temperature=0.5, contrast_mode='one', base_temperature=0.5):
        super(CaContrast_loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, label_rux=None, queue_label_mapping=None, labels=None, mask=None, features_2=None,
                labels_2=None, GPU=None):
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

        device = (torch.device ('cuda:' + str (GPU))
                  if features.is_cuda
                  else torch.device ('cpu'))
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
            # [bsz, bsz]
            # reliability_mask = reliability.unsqueeze(1).repeat(1, n_samples, 1)
            # mask = torch.zeros((labels.shape[1], labels_2.shape[1]))
            # for i in range(labels.shape[1]):
            #     for j in range(labels_2.shape[1]):
            #         if label_rux[int(labels[0][i][0])] == queue_label_mapping[int(labels_2[0][j][0])]:
            #             mask[i][j] = 1
            #
            # mask = mask.unsqueeze(0).float().to(device)
            queue_label_mapping = dict (zip (queue_label_mapping.values (), queue_label_mapping.keys ()))
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

        # mask-out self-contrast cases
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     2,
        #     torch.arange(n_samples * anchor_count).view(1, -1, 1).repeat(features.shape[0], 1, 1).to(device),
        #     0
        # )
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