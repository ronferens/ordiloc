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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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


def get_labels_for_ordinal_classification_l2_norm(labels, centroids):
    # Setting the initial labels order
    indexes = np.unique(labels, return_index=True)[1]
    org_labels_order = np.arange(len(indexes))

    # Listing all possible labels permutations
    perms = set(permutations(org_labels_order))

    # Finding the labels order that generate the largest distance between the centroids
    new_labels_order = None
    max_dist_perm = 0
    for p in tqdm(perms, desc='Setting best labels for ordinal classification'):
        distances = np.linalg.norm(centroids - centroids[p[0]], axis=1)
        total_dist = np.sum(distances)
        if total_dist > max_dist_perm:
            max_dist_perm = total_dist
            new_labels_order = distances.argsort()

    return org_labels_order, new_labels_order


def get_labels_for_ordinal_classification_lda(labels, centroids):
    # Setting the initial labels order
    indexes = np.unique(labels, return_index=True)[1]
    org_labels_order = np.array([labels[index] for index in sorted(indexes)])

    lda = LinearDiscriminantAnalysis(n_components=1)
    data_1d = lda.fit(centroids, org_labels_order.T).transform(centroids)
    new_labels_order = None

    return org_labels_order, new_labels_order


def cluster_data_for_ordinal_classification(num_of_segments, data):
    kmeans = KMeans(n_clusters=num_of_segments, random_state=0).fit(data)
    init_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Find best labels setup for ordinal-classification
    org_labels_order, new_labels_order = get_labels_for_ordinal_classification_l2_norm(init_labels, centroids)
    # org_labels_order, new_labels_order = get_labels_for_ordinal_classification_lda(init_labels, centroids)

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
    input_file = args.dataset_path
    scene = args.scene
    scene_data = {}

    for cluster_type in ['position', 'orientation']:
        data = []

        # Reading the train/test data
        scene_data['data'] = pd.read_csv(input_file)

        images_names = get_dataset_images_names(scene_data['data']['img_path'])
        num_of_imgs = len(images_names)
        scene_data['imgs'] = images_names

        # Clustering the training data and set initial labels
        if cluster_type == 'position':
            data_to_cluster = scene_data['data'][['t1', 't2', 't3']].to_numpy()
        else:
            data_to_cluster = scene_data['data'][['q1', 'q2', 'q3', 'q4']].to_numpy()
        labels = cluster_data_for_ordinal_classification(args.num_clusters, data_to_cluster)
        print(labels)

        # Visualizing only for positional clusters (using X/Y coordinates)
        for indx, label in enumerate(np.unique(labels)):
            indices = label == labels
            data.append(go.Scatter(x=scene_data['data']['t1'][indices].to_numpy(),
                                   y=scene_data['data']['t2'][indices].to_numpy(),
                                   mode='markers',
                                   marker=dict(line=dict(color='DarkSlateGrey', width=1)),
                                   name='cluster #{}'.format(indx),
                                   text=list(map(lambda fn: f'File: ' + fn, images_names))))

        scene_name_with_label = ['{}{}'.format(scene, i) for i in labels]
        scene_data['data']['scene'] = scene_name_with_label

        if args.viz:
            layout = go.Layout(title='Scene Data: <b>{}/{} - {} Segments - {}</b>'.format(args.dataset_name.title(),
                                                                                          scene,
                                                                                          args.num_clusters,
                                                                                          cluster_type.title()),
                               xaxis=dict(title='X Coordinate'),
                               yaxis=dict(title='Y Coordinate'))

            save_path = r'{}_{}_{}_segments_{}.html'.format(args.dataset_name, scene, args.num_clusters, cluster_type)
            plotly.offline.plot({'data': data, 'layout': layout}, filename=save_path, auto_open=True)
