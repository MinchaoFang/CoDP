
import pandas as pd
import numpy as np
from sympy import sequence
from transformers.models.esm.modeling_esm import EsmForMaskedLM
from transformers import AutoTokenizer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import pickle
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
import random


import os

import shutil



def copy_current_file_to_folder(destination_folder):
    """
    将当前运行的 Python 文件复制到指定的文件夹。

    参数:
        destination_folder (str): 目标文件夹路径。
    """
    # get the current file path

    current_file = os.path.abspath(__file__)  

    # check if destination folder exists, if not, create it

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # mkdir destination file path

    destination_path = os.path.join(destination_folder, os.path.basename(current_file))

    # copy this script to the destination folder

    shutil.copy2(current_file, destination_path)
    print(f"copy to: {destination_path}")


class CroppedLengthBucketedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, length_bucket_width=32, crop_size=256):
        """
        Creates a batch sampler that groups sequences into buckets based on their lengths for efficient processing.
        Sequences longer than the specified crop size are treated as having the crop size length for bucketing.

        Args:
            lengths (List[int]): A list containing the length of each sequence in the dataset.
            batch_size (int): Target number of samples per batch.
            length_bucket_width (int): Width of each length bucket. Sequences with similar lengths (within this range) will be grouped together.
            crop_size (int): Maximum sequence length threshold. Longer sequences will be truncated and treated as having this length for bucketing purposes.
        """
        self.original_lengths = lengths
        # crop lengths, treating lengths exceeding crop_size as crop_size
        self.lengths = [min(length, crop_size) for length in lengths]
        self.batch_size = batch_size
        self.length_bucket_width = length_bucket_width
        self.crop_size = crop_size
        self.indices = list(range(len(self.lengths)))
        
        # set up buckets
        self.buckets = self._create_buckets()
        # prepare batches
        self.batches = self._create_batches()
        
    def _create_buckets(self):
        """divide indices into buckets based on their lengths"""
        buckets = {}
        for idx in self.indices:
            bucket_idx = self.lengths[idx] // self.length_bucket_width
            if bucket_idx not in buckets:
                buckets[bucket_idx] = []
            buckets[bucket_idx].append(idx)
        return buckets
    
    def _create_batches(self):
        """set up batches from the buckets"""
        batches = []
        
        # reindex buckets to ensure consistent order
        for bucket_idx in sorted(self.buckets.keys()):
            bucket_indices = self.buckets[bucket_idx]
            
            # optionally shuffle within the bucket

            #random.shuffle(bucket_indices)
            
            # split into batches
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i + self.batch_size]
                if batch:  # 确保批次非空
                    batches.append(batch)
        random.shuffle(batches)
        return batches
    
    def __iter__(self):
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)
    

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss

def distogram_loss(
    logits,
    pseudo_beta,
    pseudo_beta_mask,
    bins_setting,
    eps=1e-6,
    **kwargs,
):
    bins_setting
    boundaries = torch.linspace(
        bins_setting['first_break'],
        bins_setting['last_break'],
        bins_setting['num_bins'] - 1,
        device=logits.device,
    )

    boundaries = boundaries ** 2

    dists = torch.sum(
        (pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, bins_setting['num_bins']),
    )

    square_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    # Average over the batch dimensions
    mean = torch.mean(mean)

    return mean

def distogram(
    pseudo_beta,
    pseudo_beta_mask,
    eps=1e-6,
    **kwargs,
):
    bins_setting = {
        'first_break': 2.3125,
        'last_break': 21.6875,
        'num_bins': 64
    }
    boundaries = torch.linspace(
        bins_setting['first_break'],
        bins_setting['last_break'],
        bins_setting['num_bins'] - 1,
        device=pseudo_beta.device,
    )

    boundaries = boundaries

    dists = torch.sum(
        (pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)

    square_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]
    bins = bins_setting['num_bins']
    true_bins_one_hot = torch.nn.functional.one_hot(true_bins, num_classes=bins).float()  # 形状 (B, L, L, bins)

    true_bins_one_hot *= square_mask[..., None]  # 广播到 (B, L, L, bins)


    return true_bins_one_hot, square_mask



def clean_backbone_coords(coords):
    # convert numpy.object_ to float32 array
    if coords.dtype == np.object_:
        try:
            # convert each element to float32
            return np.array([np.array(coord, dtype=np.float32) for coord in coords], dtype=np.float32)
        except Exception as e:
            print(f"error: {e}")
            return None
    return coords  
  

class MaximizeCosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(MaximizeCosineSimilarityLoss, self).__init__()

    def forward(self, z_i, z_j):
        """
        输入:
        - z_i: 正样本的 embedding，形状 [batch_size, embedding_dim]
        - z_j: 与 z_i 对应的正样本的 embedding，形状 [batch_size, embedding_dim]

        输出:
        - loss: 通过最大化余弦相似度计算的损失

        """
        # 对 embedding 进行 L2 归一化

        z_i = F.normalize(z_i, p=2, dim=1)  # 归一化到单位向量

        z_j = F.normalize(z_j, p=2, dim=1)

        # 计算每对样本的余弦相似度

        cosine_sim = F.cosine_similarity(z_i, z_j, dim=1)  # 计算每个样本对的余弦相似度

        loss = cosine_sim.mean()

        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss implementation
        
        Args:
        - alpha: balancing parameter
        - gamma: focusing parameter
        - reduction: reduction method
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets, mask=None):
        """
        Args:
        - inputs: model predictions (B, C, H, W)
        - targets: ground truth contact map (B, C, H, W)
        - mask: optional mask for valid regions
        """
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if mask is not None:
            F_loss = F_loss * mask.unsqueeze(1).float()
            if self.reduction == 'mean':
                return F_loss.sum() / (mask.sum() * targets.size(1) + 1e-10)
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
def pairwise_ranking_loss(model_values, target_values):
    """
    Computes the Pairwise Ranking Loss to measure the consistency between model's ranking and the target ranking.

    This loss function evaluates whether the relative ordering of samples by the model's predictions
    matches the ordering defined by the target reference values.

    Args:
        model_values (list[float]): Values computed by the model used for ranking.
        target_values (list[float]): Target reference values (e.g., plDDT, RMSD) that define the ideal ranking.

    Returns:
        float: The computed ranking loss value. Lower values indicate better alignment with the target ranking.
    """
    # convert to numpy arrays for easier manipulation

    model_values = np.array(model_values)
    target_values = np.array(target_values)
    
    # initialize loss

    loss = 0.0

    
    # double loop over all pairs

    n = len(model_values)
    for i in range(n):
        for j in range(i + 1, n):
            # ranking difference

            model_diff = model_values[i] - model_values[j]
            target_diff = target_values[i] - target_values[j]
            
            # accumulate loss if rankings mismatch

            if model_diff * target_diff < 0:  # ranking mismatch

                loss += 1

    # normalize the loss by the number of pairs

    loss /= (n * (n - 1) / 2)  # pairwise combinations

    
    return loss
