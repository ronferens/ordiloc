import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdiPoseNet(nn.Module):
    """
    A class to represent a classic pose regressor (PoseNet) with an efficient-net backbone
    PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization,
    Kendall et al., 2015
    """
    def __init__(self, config, backbone_path):
        """
        Constructor
        :param backbone_path: backbone path to a resnet backbone
        """
        super(OrdiPoseNet, self).__init__()

        # Loading the backbone and setting the matching configuration
        self.backbone = torch.load(backbone_path)
        backbone_dim = config.get('backbone_dim')
        latent_dim = config.get('latent_dim')
        num_classes_pos = config.get('num_classes_pos')
        num_classes_orient = config.get('num_classes_orient')

        # Regressor layers
        self.fc1 = nn.Linear(backbone_dim, latent_dim)
        self.cls_pos = nn.Linear(latent_dim, num_classes_pos)
        self.cls_orient = nn.Linear(latent_dim, num_classes_orient)

        # Creating the regression heads
        self.cls_pos_fc = nn.Linear(num_classes_pos, latent_dim)
        self.cls_orient_fc = nn.Linear(num_classes_orient, latent_dim)

        self.reg_pos = nn.Linear(latent_dim, 3)
        self.reg_orient = nn.Linear(latent_dim, 4)

        self.dropout = nn.Dropout(p=0.1)
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

        # Centroids
        self._cent_pos = None
        self._cent_orient = None

    @staticmethod
    def convert_pred_to_label(pred):
        label = F.relu(torch.sum(torch.cumprod((pred > 0.5), dim=1), dim=1) - 1)
        return label.to(dtype=torch.int64)

    def set_centroids(self, cent_pos, cent_orient):
        self._cent_pos = cent_pos
        self._cent_orient = cent_orient

    def forward(self, data):
        """
        Forward pass
        :param data: (torch.Tensor) dictionary with key-value 'img' -- input image (N X C X H X W)
        :return: (torch.Tensor) dictionary with key-value 'pose' -- 7-dimensional absolute pose for (N X 7)
        """
        if hasattr(self.backbone, 'extract_features'):
            x = self.backbone.extract_features(data.get('img'))
        else:
            x = self.backbone(data.get('img'))

        x = self.avg_pooling_2d(x)
        x = x.flatten(start_dim=1)

        x = self.dropout(F.relu(self.fc1(x)))

        # Performing the ordinal classification
        cls_x = F.sigmoid(self.cls_pos(x))
        cls_q = F.sigmoid(self.cls_orient(x))

        # Regressing the camera pose
        # cls_x_latent = F.relu(self.cls_pos_fc(cls_x))
        # cls_q_latent = F.relu(self.cls_orient_fc(cls_q))

        p_x = self.reg_pos(x) + self._cent_pos[self.convert_pred_to_label(cls_x)]
        p_q = self.reg_orient(x) + self._cent_orient[self.convert_pred_to_label(cls_q)]

        return {'pose': torch.cat((p_x, p_q), dim=1), 'pose_cls': torch.stack((cls_x, cls_q), dim=2)}

