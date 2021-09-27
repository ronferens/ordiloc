import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join, splitext
import argparse


def get_segments_centroids(labels, data):
    num_segments = np.unique(labels)
    centroids = np.zeros((num_segments.shape[0], data.shape[1]))
    for label in np.unique(labels):
        indices = label == labels
        centroids[label, :] = np.mean(data[indices], axis=0)
    return centroids


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset_name', help='The dataset name', type=str)
    arg_parser.add_argument('dataset_path', help='The path to the train dataset .csv file', type=str)
    arg_parser.add_argument('scene', help='The name of the scene to cluster', type=str)
    arg_parser.add_argument('--num_bins', help='The number of bins to use for the analysis histogram', type=int,
                            default=100)
    args = arg_parser.parse_args()

    # Setting input scene and path
    # ============================
    input_file = args.dataset_path
    dataset_name = args.dataset_name
    scene = args.scene

    # Reading the train and test data
    scene_data = pd.read_csv(input_file)

    for cluster_type in ['position', 'orientation']:
        data = []

        # Clustering the training data and set initial labels
        if cluster_type == 'position':
            data_to_cluster = ['t1', 't2', 't3']
            clusters_labels = 'class_position'
        else:
            data_to_cluster = ['q1', 'q2', 'q3', 'q4']
            clusters_labels = 'class_orientation'

        # Retrieving the training data centroids
        labels = scene_data[clusters_labels].to_numpy()
        centroids = get_segments_centroids(labels, scene_data[data_to_cluster].values)
        num_clusters = centroids.shape[0]

        # Visualizing only for positional clusters (using X/Y coordinates)
        distances = []
        for label in np.unique(labels):
            indices = label == scene_data[clusters_labels].values
            cluster_data = scene_data[data_to_cluster][indices].values

            if cluster_type == 'position':
                distances += np.linalg.norm(cluster_data - centroids[label], axis=1).tolist()
                err_units_str = 'meters'
            else:
                import torch.nn.functional as F
                import torch
                est_pose_q = F.normalize(torch.tensor(cluster_data), p=2, dim=1)
                gt_pose_q = F.normalize(torch.tensor(centroids[label]).expand(cluster_data.shape[0], 4), p=2, dim=1)
                inner_prod = torch.bmm(est_pose_q.view(est_pose_q.shape[0], 1, est_pose_q.shape[1]),
                                       gt_pose_q.view(gt_pose_q.shape[0], gt_pose_q.shape[1], 1))
                orient_err = 2 * torch.acos(torch.abs(inner_prod)) * 180 / np.pi
                distances += orient_err.squeeze().data.cpu().numpy().tolist()
                err_units_str = 'degrees'

        mean_clustering_err = np.mean(distances)
        std_clustering_err = np.std(distances)

        plt.figure()
        plt.hist(distances, args.num_bins, density=True, alpha=0.75)
        plt.axvline(x=mean_clustering_err, color='r',
                    label=r'$Err_{Mean}=$' + '{:.3f} [{}]'.format(mean_clustering_err, err_units_str))
        plt.xlabel('Clustering error [{}]'.format(err_units_str))
        plt.ylabel('Distribution')
        plt.title('Clustering error - {}\n{} - {} - {} Segments\n'.format(cluster_type.title(), dataset_name, scene,
                                                                          num_clusters) +
                  ' $Err_{Mean}=$' + '{:.3f} [{}]'.format(mean_clustering_err, err_units_str) +
                  '\t$Err_{STD}=$' + '{:.3f}'.format(std_clustering_err))
        plt.legend()
        plt.grid()
        plt.show()
