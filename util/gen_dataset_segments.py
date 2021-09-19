import numpy as np
import pandas as pd
from os import listdir
from os.path import join, isfile, exists
from tqdm import tqdm
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
from sklearn.cluster import KMeans
import argparse
from itertools import permutations

init_notebook_mode(connected=False)


def verify_input_file(path):
    if not exists(path):
        print('File not found')
        exit(-1)
    else:
        print('Found file: {}'.format(path))


def get_dataset_images_names(paths):
    names = []
    for img_filename in tqdm(paths):
        names.append('/'.join(img_filename.split('/')[-2:]))
    return names


def get_labels_for_ordinal_classification(labels, centroids):
    # Setting the initial labels order
    indexes = np.unique(labels, return_index=True)[1]
    org_labels_order = [labels[index] for index in sorted(indexes)]

    # Listing all possible labels permutations
    perms = set(permutations(org_labels_order))

    # Finding the labels order that generate the largest distance between the centroids
    new_labels_order = None
    max_dist_perm = 0
    for p in perms:

        distances = np.linalg.norm(centroids - centroids[p[0]], axis=1)
        total_dist = np.sum(distances)
        if total_dist > max_dist_perm:
            max_dist_perm = total_dist
            new_labels_order = distances.argsort()

    return org_labels_order, new_labels_order


def cluster_data_for_ordinal_classification(num_of_segments, data):
    kmeans = KMeans(n_clusters=num_of_segments, random_state=0).fit(data)
    init_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Find best labels setup for ordinal-classification
    org_labels_order, new_labels_order = get_labels_for_ordinal_classification(init_labels, centroids)

    # Assigning the new labels
    new_labels = np.zeros_like(init_labels)
    for idx, l in enumerate(org_labels_order):
        indices = l == init_labels
        new_labels[indices] = new_labels_order[idx]

    return new_labels


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset_name', help='The dataset name', type=str)
    arg_parser.add_argument('dataset_path', help='The path to the dataset .csv file', type=str)
    arg_parser.add_argument('scene', help='The name of the scene to cluster', type=str)
    arg_parser.add_argument('num_clusters', help='Number of clusters in the dataset', type=int)
    arg_parser.add_argument('--viz', help='Indicates whether to visualize the positional clustering',
                            action='store_true', default=False)
    args = arg_parser.parse_args()

    # Setting input scene and path
    # ============================
    path = args.dataset_path
    scene = args.scene

    types = ['train', 'test']
    scene_data = {'train': {}, 'test': {}}
    data = []
    for data_type in types:
        # Retrieving input files (train and test)
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        for f in files:
            if scene in f and data_type in f:
                input_file = f
        verify_input_file(input_file)

        # Reading the train/test data
        scene_data[data_type]['data'] = pd.read_csv(input_file)

        images_names = get_dataset_images_names(scene_data[data_type]['data']['img_path'])
        num_of_imgs = len(images_names)
        scene_data[data_type]['imgs'] = images_names

        labels = {}
        if data_type == 'train':
            for cluster_type in ['position', 'orientation']:
                # Clustering the training data and set initial labels
                if cluster_type == 'position':
                    data_to_cluster = scene_data[data_type]['data'][['t1', 't2', 't3']].to_numpy()
                else:
                    data_to_cluster = scene_data[data_type]['data'][['q1', 'q2', 'q3', 'q4']].to_numpy()
                labels[cluster_type] = cluster_data_for_ordinal_classification(args.num_clusters, data_to_cluster)

                if cluster_type == 'position':
                    # Visualizing only for positional clusters (using X/Y coordinates)
                    for indx, label in enumerate(np.unique(labels[cluster_type])):
                        indices = label == labels[cluster_type]
                        data.append(go.Scatter(x=scene_data[data_type]['data']['t1'][indices].to_numpy(),
                                               y=scene_data[data_type]['data']['t2'][indices].to_numpy(),
                                               mode='markers',
                                               name='{} cluster #{}'.format(data_type.title(), indx),
                                               text=list(map(lambda fn: f'File: ' + fn, images_names))))

                    scene_name_with_label = ['{}{}'.format(scene, i) for i in labels[cluster_type]]
                    scene_data[data_type]['data']['scene'] = scene_name_with_label
        else:
            # Plotting the test data (black dots)
            data.append(go.Scatter(x=scene_data[data_type]['data']['t1'].to_numpy(),
                                   y=scene_data[data_type]['data']['t2'].to_numpy(),
                                   mode='markers',
                                   marker=dict(size=8, color='black'),
                                   name='{} Data'.format(data_type.title()),
                                   text=list(map(lambda fn: f'File: ' + fn, images_names))))

    if args.viz:
        layout = go.Layout(title='Scene Data: <b>{}/{} - {} Segments</b>'.format(args.dataset_name.title(), scene,
                                                                                 args.num_clusters),
                           xaxis=dict(title='X Coordinate'),
                           yaxis=dict(title='Y Coordinate'))

        save_path = r'scene_train_data_plot_{}_{}.html'.format(args.dataset_name, scene)
        plotly.offline.plot({'data': data, 'layout': layout}, filename=save_path, auto_open=True)
