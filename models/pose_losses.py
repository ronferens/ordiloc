import torch
import torch.nn.functional as F
import torch.nn as nn
from util import utils


class CameraPoseLoss(nn.Module):
    """
    A class to represent camera pose loss
    """

    def __init__(self, config):
        """
        :param config: (dict) configuration to determine behavior
        """
        super(CameraPoseLoss, self).__init__()
        self.learnable = config.get("learnable")
        self.s_x = torch.nn.Parameter(torch.Tensor([config.get("s_x")]), requires_grad=self.learnable)
        self.s_q = torch.nn.Parameter(torch.Tensor([config.get("s_q")]), requires_grad=self.learnable)
        self.norm = config.get("norm")

    def forward(self, est_pose, gt_pose):
            """
            Forward pass
            :param est_pose: (torch.Tensor) batch of estimated poses, a Nx7 tensor
            :param gt_pose: (torch.Tensor) batch of ground_truth poses, a Nx7 tensor
            :return: camera pose loss
            """
            # Position loss
            l_x = torch.norm(gt_pose[:, 0:3] - est_pose[:, 0:3], dim=1, p=self.norm).mean()
            # Orientation loss (normalized to unit norm)
            l_q = torch.norm(F.normalize(gt_pose[:, 3:], p=2, dim=1) - F.normalize(est_pose[:, 3:], p=2, dim=1),
                             dim=1, p=self.norm).mean()

            if self.learnable:
                return l_x * torch.exp(-self.s_x) + self.s_x + l_q * torch.exp(-self.s_q) + self.s_q
            else:
                return self.s_x*l_x + self.s_q*l_q


class CameraPoseOrdinalLoss(nn.Module):
    """
    A class to represent camera pose classification loss
    """

    def __init__(self, config):
        """
        :param config: (dict) configuration to determine behavior
        """
        super(CameraPoseOrdinalLoss, self).__init__()
        self.inter_loss_weight = config.get("inter_loss_weight")

    def forward(self, est_pose_class, gt_labels):
            """
            Forward pass
            :param est_pose_class: (torch.Tensor) batch of estimated poses classes, a NxNumOfSegmentsx2 tensor
            :param gt_labels: (torch.Tensor) batch of ground_truth poses classes, a Nx2 tensor
            :return: camera pose loss
            """
            gt_pose_class = utils.convert_pose_labels_to_classes(est_pose_class.shape[1], gt_labels)

            ordi_loss = self.inter_loss_weight * nn.MSELoss(reduction='none')(est_pose_class[:, :, 0], gt_pose_class[:, :, 0]).sum(axis=1) + \
                        (1 - self.inter_loss_weight) * nn.MSELoss(reduction='none')(est_pose_class[:, :, 1], gt_pose_class[:, :, 1]).sum(axis=1)
            ordi_loss = torch.sum(ordi_loss)

            return ordi_loss
