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

init_notebook_mode(connected=False)


def verify_input_file(path):
    if not exists(path):
        print('File not found')
        exit(-1)
    else:
        print('Found file: {}'.format(path))


def get_dataset_images_names(paths):
    images_names = []
    for img_filename in tqdm(paths):
        images_names.append('/'.join(img_filename.split('/')[-2:]))
    return images_names


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset_name', help='The dataset name', type=str)
    arg_parser.add_argument('dataset_path', help='The path to the dataset .csv file', type=str)
    arg_parser.add_argument('scene', help='The name of the scene to cluster', type=str)
    arg_parser.add_argument('num_clusters', help='Number of clusters in the dataset', type=int)
    args = arg_parser.parse_args()

    # Setting input scene and path
    # ============================
    path = args.dataset_path
    scene = args.scene

    types = ['train', 'test']
    scene_data = {'train': {}, 'test': {}}
    data = []
    for data_type in types:
        # Retrieving input files
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

        if data_type == 'train':
            # Clustering the training data
            num_of_segments = args.num_clusters
            gt_pose = scene_data[data_type]['data'][['t1', 't2', 't3', 'q1', 'q2', 'q3', 'q4']].to_numpy()
            kmeans = KMeans(n_clusters=num_of_segments, random_state=0).fit(gt_pose)
            labels = kmeans.labels_
            for indx, label in enumerate(np.unique(labels)):
                indices = label == labels
                data.append(go.Scatter(x=scene_data[data_type]['data']['t1'][indices].to_numpy(),
                                       y=scene_data[data_type]['data']['t2'][indices].to_numpy(),
                                       mode='markers',
                                       name='{} cluster #{}'.format(data_type.title(), indx),
                                       text=list(map(lambda fn: f'File: ' + fn, images_names))))

            scene_name_with_label = ['{}{}'.format(scene, i) for i in labels]
            scene_data[data_type]['data']['scene'] = scene_name_with_label
            # scene_data[data_type]['data'].to_csv(join(data_root_dir, input_file.replace('_train.', '_MS_train.')))
        else:
            # Plotting the test data (black dots)
            data.append(go.Scatter(x=scene_data[data_type]['data']['t1'].to_numpy(),
                                   y=scene_data[data_type]['data']['t2'].to_numpy(),
                                   mode='markers',
                                   marker=dict(size=8, color='black'),
                                   name='{} Data'.format(data_type.title()),
                                   text=list(map(lambda fn: f'File: ' + fn, images_names))))

    layout = go.Layout(title='Scene Data: <b>{}/{} - {} Segments</b>'.format(args.dataset_name.title(), scene, num_of_segments),
                       xaxis=dict(title='X Coordinate'),
                       yaxis=dict(title='Y Coordinate'))

    save_path = r'scene_train_data_plot_{}_{}.html'.format(args.dataset_name, scene)
    plotly.offline.plot({'data': data, 'layout': layout}, filename=save_path, auto_open=True)
