import torch
import torch.nn.functional as F

def find_closest_indices(A, B):
    dist_matrix = torch.cdist(A, B, p=2)
    closest_indices = torch.argmin(dist_matrix, dim=1)
    used_indices = closest_indices.tolist()
    return closest_indices, used_indices

def match_samples(means_src, means_tar, values_src, values_tar):
    # 将means_src中元素的标签与数组means_tar中的标签进行比较
    mask = torch.tensor([label in values_tar for label in values_src], dtype=torch.bool)

    # 创建一个存储正样本的tensor
    positive_samples = torch.zeros_like(means_src)
    positive_indices = torch.zeros(len(means_src), dtype=torch.long)

    # 为标签在means_tar中的元素分配最近的正样本
    if mask.any():
        closest_indices, used_indices = find_closest_indices(means_src[mask], means_tar)
        positive_samples[mask] = means_tar[closest_indices]
        closest_indices = closest_indices.to(positive_indices.device)
        positive_indices[mask] = closest_indices
    else:
        used_indices = []

    # 找到means_src中其他元素
    remaining_indices_src = torch.arange(len(means_src))[~mask]

    # 找到means_tar中未被选择的元素
    remaining_indices_tar = torch.arange(len(means_tar))
    remaining_indices_tar = remaining_indices_tar[~torch.isin(remaining_indices_tar, torch.tensor(used_indices))]

    # 按顺序选择剩余的means_tar中的元素作为正样本
    if len(remaining_indices_src) <= len(remaining_indices_tar):
        selected_indices = remaining_indices_tar[:len(remaining_indices_src)]
    else:
        raise ValueError("Not enough remaining elements in B to match remaining elements in A")

    positive_samples[remaining_indices_src] = means_tar[selected_indices]
    positive_indices[remaining_indices_src] = selected_indices

    # 获取负样本
    negative_samples = []
    for idx in range(len(means_src)):
        negative_indices = torch.cat((torch.arange(0, idx), torch.arange(idx + 1, len(means_src))))
        negative_samples.append(means_src[negative_indices])

    negative_samples = torch.stack(negative_samples)

    return positive_samples, negative_samples

def simclr_loss(anchors, positives, negatives, temperature=0.6):
    batch_size = anchors.size(0)
    embedding_size = anchors.size(1)
    num_negatives = negatives.size(1)

    # L2 归一化
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # 计算锚点与正样本的余弦相似度
    positive_sim = F.cosine_similarity(anchors, positives, dim=-1) / temperature

    # 计算锚点与负样本的余弦相似度
    anchors_expanded = anchors.view(batch_size, 1, embedding_size).expand(-1, num_negatives, -1)
    negative_sim = F.cosine_similarity(anchors_expanded, negatives, dim=-1) / temperature

    # 拼接正样本和负样本的相似度
    logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long).to(anchors.device)

    # 计算SimCLR loss
    loss = F.cross_entropy(logits, labels)

    return loss
