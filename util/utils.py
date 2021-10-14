import logging
import logging.config
import PIL
import json
from os.path import join, exists, split, realpath
import time
from os import mkdir, getcwd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd

# Logging and output utils
##########################
def get_stamp_from_log():
    """
    Get the time stamp from the log file
    :return:
    """
    return split(logging.getLogger().handlers[0].baseFilename)[-1].replace(".log","")


def create_output_dir(name):
    """
    Create a new directory for outputs, if it does not already exist
    :param name: (str) the name of the directory
    :return: the path to the outpur directory
    """
    out_dir = join(getcwd(), name)
    if not exists(out_dir):
        mkdir(out_dir)
    return out_dir


def init_logger():
    """
    Initialize the logger and create a time stamp for the file
    """
    path = split(realpath(__file__))[0]

    with open(join(path, 'log_config.json')) as json_file:
        log_config_dict = json.load(json_file)
        filename = log_config_dict.get('handlers').get('file_handler').get('filename')
        filename = ''.join([filename, "_", time.strftime("%d_%m_%y_%H_%M", time.localtime()), ".log"])

        # Creating logs' folder is needed
        log_path = create_output_dir('out')

        log_config_dict.get('handlers').get('file_handler')['filename'] = join(log_path, filename)
        logging.config.dictConfig(log_config_dict)

        # disable external modules' loggers (level warning and below)
        logging.getLogger(PIL.__name__).setLevel(logging.WARNING)


# Evaluation utils
##########################
def convert_pose_labels_to_classes(num_segments, labels):
    pose_class = torch.zeros((labels.shape[0], num_segments, 2)).to(labels.device)

    for i, target in enumerate(labels):
        pose_class[i, 0:(target[0].int() + 1), 0] = 1
        pose_class[i, 0:(target[1].int() + 1), 1] = 1

    return pose_class


def pose_err(est_pose, gt_pose):
    """
    Calculate the position and orientation error given the estimated and ground truth pose(s
    :param est_pose: (torch.Tensor) a batch of estimated poses (Nx7, N is the batch size)
    :param gt_pose: (torch.Tensor) a batch of ground-truth poses (Nx7, N is the batch size)
    :return: position error(s) and orientation errors(s)
    """
    posit_err = torch.norm(est_pose[:, 0:3] - gt_pose[:, 0:3], dim=1)
    est_pose_q = F.normalize(est_pose[:, 3:], p=2, dim=1)
    gt_pose_q = F.normalize(gt_pose[:, 3:], p=2, dim=1)
    inner_prod = torch.bmm(est_pose_q.view(est_pose_q.shape[0], 1, est_pose_q.shape[1]),
                           gt_pose_q.view(gt_pose_q.shape[0], gt_pose_q.shape[1], 1))
    orient_err = 2 * torch.acos(torch.abs(inner_prod)) * 180 / np.pi
    return posit_err, orient_err


def convert_pred_to_label(pred):
    labels = torch.nn.functional.relu((pred > 0.5).cumprod(axis=1).sum(axis=1) - 1)
    return labels


def pose_class_err(preds, gt_labels):
    """
    Calculate the position and orientation error given the estimated and ground truth pose(s
    :param preds: (torch.Tensor) a batch of estimated poses (Nx7, N is the batch size)
    :param gt_pose: (torch.Tensor) a batch of ground-truth poses (Nx7, N is the batch size)
    :return: position error(s) and orientation errors(s)
    """
    est_position_class = convert_pred_to_label(preds[:, :, 0])
    est_orientation_class = convert_pred_to_label(preds[:, :, 1])

    pose_class_err = est_position_class == gt_labels[:, 0]
    orient_class_err = est_orientation_class == gt_labels[:, 1]
    return pose_class_err, orient_class_err


def load_clusters_centroids(input_file, device):
    scene_data = pd.read_csv(input_file)

    num_clusters = np.unique(scene_data['class_position'].to_numpy()).shape[0]
    cent_pos = torch.zeros((num_clusters, 3)).to(device)
    for l in range(num_clusters):
        for i in range(3):
            cent_pos[l, i] = scene_data['cent_position_{}'.format(i + 1)][scene_data['class_position'] == l].iloc[0]

    num_clusters = np.unique(scene_data['class_orientation'].to_numpy()).shape[0]
    cent_orient = torch.zeros((num_clusters, 4)).to(device)
    for l in range(num_clusters):
        for i in range(4):
            cent_orient[l, i] = scene_data['cent_orientation_{}'.format(i + 1)][scene_data['class_orientation'] == l].iloc[0]

    return cent_pos, cent_orient


# Plotting utils
##########################
def plot_loss_func(sample_count, loss_vals, loss_fig_path):
    plt.figure()
    plt.plot(sample_count, loss_vals)
    plt.grid()
    plt.title('Camera Pose Loss')
    plt.xlabel('Number of samples')
    plt.ylabel('Loss')
    plt.savefig(loss_fig_path)


# Augmentations
train_transforms = {
    'baseline': transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

}
test_transforms = {
    'baseline': transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
        ])
}
