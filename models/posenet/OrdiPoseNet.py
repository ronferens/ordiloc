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
        num_classes = config.get('num_classes')

        # Regressor layers
        self.fc1 = nn.Linear(backbone_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, num_classes)
        self.fc3 = nn.Linear(latent_dim, num_classes)

        self.dropout = nn.Dropout(p=0.1)
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

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
        c_x = F.sigmoid(self.fc2(x))
        c_q = F.sigmoid(self.fc3(x))

        return {'pose': torch.stack((c_x, c_q), dim=2)}

