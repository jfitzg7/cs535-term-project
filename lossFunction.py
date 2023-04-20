invariance_loss_weight: float = 25.0
variance_loss_weight: float = 25.0
covariance_loss_weight: float = 1.0
variance_loss_epsilon: float = 1e-04
import torch
import torch.nn as nn

import torch.nn.functional as F
# https://generallyintelligent.com/open-source/2022-04-21-vicreg/

class WeightedCovarianceLoss(nn.Module):
    def __init__(self):
        super(WeightedCovarianceLoss, self).__init__()

    def forward(self, output, target):
        # may not be best 
        all_loss = 0
        for o,t in zip(output,target):
            #print('output',o[0].shape)
            #print('target',t[0].shape)
            losses =  self.get_vicreg_loss(o[0],t[0])
            all_loss += losses['loss']
        return all_loss/len(output)

    def get_vicreg_loss(self, z_a, z_b):
       # print('output',z_a.shape)
        #print('target',z_b.shape)
        assert z_a.shape == z_b.shape and len(z_a.shape) == 2

        # invariance loss
        loss_inv = F.mse_loss(z_a, z_b)

        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + variance_loss_epsilon)
        std_z_b = torch.sqrt(z_b.var(dim=0) + variance_loss_epsilon)
        loss_v_a = torch.mean(F.relu(1 - std_z_a))
        loss_v_b = torch.mean(F.relu(1 - std_z_b))
        loss_var = loss_v_a + loss_v_b

        # covariance loss
        N, D = z_a.shape
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = ((z_a.T @ z_a) / (N - 1)).square()  # DxD
        cov_z_b = ((z_b.T @ z_b) / (N - 1)).square()  # DxD
        loss_c_a = (cov_z_a.sum() - cov_z_a.diagonal().sum()) / D
        loss_c_b = (cov_z_b.sum() - cov_z_b.diagonal().sum()) / D
        loss_cov = loss_c_a + loss_c_b

        weighted_inv = loss_inv * invariance_loss_weight
        weighted_var = loss_var * variance_loss_weight
        weighted_cov = loss_cov * covariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov

        return {
            "loss": loss,
            "loss_invariance": weighted_inv,
            "loss_variance": weighted_var,
            "loss_covariance": weighted_cov,
        }

class CosineDistanceLoss(nn.Module):
    def __init__(self):
        super(CosineDistanceLoss, self).__init__()

    def forward(self, output, target):
        # may not be best 
        all_loss = 0
        # print(output,target)
    
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        loss = cos(output, target)
       
        # print('output',o.shape)
        #     print('target',t.shape)
        #     losses =  cos(output, target)
        #     all_loss += losses
        return -loss