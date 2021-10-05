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


def get_labels_for_ordinal_classification(num_of_segments, data):
    # Performing PCA to project the data into 2D array
    pca = PCA(n_components=2)
    data_pca = pca.fit(data).transform(data)

    # Clustering the 2D projected data and sorting by centroids
    kmeans = KMeans(n_clusters=num_of_segments, random_state=0).fit(data_pca)
    centroids = kmeans.cluster_centers_

    # Debug visualization of segmented clusters
    # -----------------------------------------
    # import matplotlib.pyplot as plt
    # plt.figure()
    # for i in range(num_of_segments):
    #     indices = i == kmeans.labels_
    #     plt.scatter(data_pca[indices, 0], data_pca[indices, 1], label='label {}'.format(i))
    #     plt.scatter(centroids[i, 0], centroids[i, 1], label='centroid {}'.format(i))
    # plt.legend()
    # plt.grid()
    # plt.show()

    # @TODO: Consider LDA instead of the 1D PCA
    pca_centroids = PCA(n_components=1)
    centroids_pca = pca_centroids.fit(centroids).transform(centroids)
    org_labels_order = np.argsort(centroids_pca.reshape(1, -1))[0]

    # Assigning the new labels
    new_labels_order = np.arange(num_of_segments)
    new_labels = np.zeros_like(kmeans.labels_)
    for new_l, l in enumerate(org_labels_order):
        indices = l == kmeans.labels_
        new_labels[indices] = new_labels_order[new_l]

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

    # Reading the train data
    scene_data = pd.read_csv(input_file)

    for cluster_type in ['position', 'orientation']:
        data = []

        images_names = get_dataset_images_names(scene_data['img_path'])
        num_of_imgs = len(images_names)

        # Clustering the training data and set initial labels
        if cluster_type == 'position':
            data_to_cluster = scene_data[['t1', 't2', 't3']].to_numpy()
        else:
            data_to_cluster = scene_data[['q1', 'q2', 'q3', 'q4']].to_numpy()

        labels = get_labels_for_ordinal_classification(args.num_clusters, data_to_cluster)

        # Visualizing only for positional clusters (using X/Y coordinates)
        for label in np.unique(labels):
            indices = label == labels
            data.append(go.Scatter(x=scene_data['t1'][indices].to_numpy(),
                                   y=scene_data['t2'][indices].to_numpy(),
                                   mode='markers',
                                   marker=dict(line=dict(color='DarkSlateGrey', width=1)),
                                   name='cluster #{}'.format(label),
                                   text=list(map(lambda fn: f'File: ' + fn, images_names))))

        # Adding the labels to the dataset data
        scene_data['class_{}'.format(cluster_type)] = labels

        if args.viz:
            layout = go.Layout(title='Scene Data: <b>{}/{} - {} Segments - {}</b>'.format(args.dataset_name.title(),
                                                                                          scene,
                                                                                          args.num_clusters,
                                                                                          cluster_type.title()),
                               xaxis=dict(title='X Coordinate'),
                               yaxis=dict(title='Y Coordinate'))

            save_path = r'{}_{}_{}_segments_{}.html'.format(args.dataset_name, scene, args.num_clusters, cluster_type)
            plotly.offline.plot({'data': data, 'layout': layout}, filename=save_path, auto_open=True)

    # Saving the dataset data
    output_file_path = splitext(input_file)[0] + '_{}_classes'.format(args.num_clusters) + splitext(input_file)[1]
    scene_data.to_csv(output_file_path)
