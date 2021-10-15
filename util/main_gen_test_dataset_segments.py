import numpy as np
import pandas as pd
from os.path import join, splitext, exists
from tqdm import tqdm
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
from sklearn.cluster import KMeans
import argparse
from sklearn.decomposition import PCA

init_notebook_mode(connected=False)


def verify_input_file(path):
    if not exists(path):
        print('File not found')
        exit(-1)
    else:
        print('Found file: {}'.format(path))


def get_dataset_images_names(paths):
    names = []
    for img_filename in tqdm(paths, desc='loading dataset images'):
        names.append('/'.join(img_filename.split('/')[-2:]))
    return names


def get_segments_centroids(labels, data):
    num_segments = np.unique(labels)
    centroids = np.zeros((num_segments.shape[0], data.shape[1]))
    for label in np.unique(labels):
        indices = label == labels
        centroids[label, :] = np.mean(data[indices], axis=0)
    return centroids


def assign_labels(centroids, data):
    labels = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        dist = np.linalg.norm(centroids - data[i, :], axis=1)
        labels[i] = np.argmin(dist)

    return labels


def gen_segments_colors(num_segments):
    colors = []
    r = np.random.randint(0, 256, num_segments, dtype=int)
    g = np.random.randint(0, 256, num_segments, dtype=int)
    b = np.random.randint(0, 256, num_segments, dtype=int)
    for c in range(num_segments):
        colors.append('rgb({}, {}, {})'.format(r[c], b[c], g[c]))
    return colors


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset_name', help='The dataset name', type=str)
    arg_parser.add_argument('train_dataset_path', help='The path to the train dataset .csv file', type=str)
    arg_parser.add_argument('test_dataset_path', help='The path to the test dataset .csv file', type=str)
    arg_parser.add_argument('scene', help='The name of the scene to cluster', type=str)
    arg_parser.add_argument('--viz', help='Indicates whether to visualize the positional clustering',
                            action='store_true', default=False)
    args = arg_parser.parse_args()

    # Setting input scene and path
    # ============================
    input_train_file = args.train_dataset_path
    input_test_file = args.test_dataset_path
    scene = args.scene

    # Reading the train and test data
    scene_train_data = pd.read_csv(input_train_file)
    train_images_names = get_dataset_images_names(scene_train_data['img_path'])

    scene_test_data = pd.read_csv(input_test_file)

    for cluster_type in ['position', 'orientation']:
        data = []

        images_names = get_dataset_images_names(scene_test_data['img_path'])
        num_of_imgs = len(images_names)

        # Clustering the training data and set initial labels
        if cluster_type == 'position':
            data_to_cluster = ['t1', 't2', 't3']
            clusters_labels = 'class_position'
        else:
            data_to_cluster = ['q1', 'q2', 'q3', 'q4']
            clusters_labels = 'class_orientation'

        # Retrieving the training data centroids
        centroids = get_segments_centroids(scene_train_data[clusters_labels].values,
                                           scene_train_data[data_to_cluster].to_numpy())
        num_clusters = centroids.shape[0]
        colors = gen_segments_colors(num_clusters)

        # Assigning the test data with labels
        labels = assign_labels(centroids, scene_test_data[data_to_cluster].to_numpy())

        # Visualizing only for positional clusters (using X/Y coordinates)
        for label in np.unique(labels):
            indices = label == scene_train_data[clusters_labels].values
            data.append(go.Scatter(x=scene_train_data['t1'][indices].to_numpy(),
                                   y=scene_train_data['t2'][indices].to_numpy(),
                                   mode='markers',
                                   marker=dict(color=colors[int(label)], line=dict(color='DarkSlateGrey', width=1)),
                                   name='Train cluster #{}'.format(label),
                                   text=list(map(lambda fn: f'File: ' + fn, train_images_names))))

            indices = label == labels
            data.append(go.Scatter(x=scene_test_data['t1'][indices].to_numpy(),
                                   y=scene_test_data['t2'][indices].to_numpy(),
                                   mode='markers',
                                   marker=dict(color=colors[int(label)]),
                                   name='Test cluster #{}'.format(label),
                                   text=list(map(lambda fn: f'File: ' + fn, images_names))))

        # Adding the labels to the dataset data
        scene_test_data['class_{}'.format(cluster_type)] = labels

        if args.viz:
            layout = go.Layout(title='Scene Data: <b>{}/{} - {} Segments - {}</b>'.format(args.dataset_name.title(),
                                                                                          scene,
                                                                                          num_clusters,
                                                                                          cluster_type.title()),
                               xaxis=dict(title='X Coordinate'),
                               yaxis=dict(title='Y Coordinate'))

            save_path = r'{}_{}_{}_segments_{}.html'.format(args.dataset_name, scene, num_clusters, cluster_type)
            plotly.offline.plot({'data': data, 'layout': layout}, filename=save_path, auto_open=True)

        if cluster_type == 'position':
            num_clusters_pose = num_clusters
        else:
            num_clusters_orient = num_clusters

    # Saving the dataset data
    output_file_path = splitext(input_test_file)[0] + '_{}_{}_classes'.format(num_clusters_pose, num_clusters_orient) +\
                       splitext(input_test_file)[1]
    scene_test_data.to_csv(output_file_path)
