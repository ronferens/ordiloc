import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


def calc_norm_diff(values, query_idx):
    # Calculating the distance between the current image's tensor to all other entries
    diff = torch.norm(values - values[query_idx], dim=1)

    # Normalizing the differences
    norm_diff = (diff - diff.min()) / (diff.max() - diff.min())

    return norm_diff


if __name__ == '__main__':
    """
    Downloading and saving the requested ResNet backbone to the 'models' folder
    The resnet_type argument Supports: resnet34, resnet50, resnet152
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset_file', help='The dataset file to process')
    arg_parser.add_argument('emb_file', help='Teh dataset\'s embedding file to set the negative and positive samples '
                                             '(.pth file)')
    arg_parser.add_argument('num_samples', help='The number of sample to extract', type=int)
    arg_parser.add_argument('alpha', help='Embedding weight', type=float)
    args = arg_parser.parse_args()

    # Loading the embedding file
    embd_data = torch.load(args.emb_file)

    # Loading the dataset
    data = pd.read_csv(args.dataset_file)
    pose_data = torch.tensor(data[['t1', 't2', 't3', 'q1', 'q2', 'q3', 'q4']].values).to(embd_data.device)

    # Preparing the output data structure
    samples_output = np.ones((embd_data.shape[0], args.num_samples))

    for idx, f in tqdm(enumerate(data['img_path']), desc='Processing the dataset...'):
        # Extracting the current image's embedding tensor
        img_emd = embd_data[idx]

        # Calculating the distance between the current image's embedding tensor to all other entries
        # (1) difference based on poses
        pose_norm_diff = calc_norm_diff(pose_data, idx)

        # (2) difference based on input embeddings
        emds_norm_diff = calc_norm_diff(embd_data, idx)

        weigthed_diff = (1 - args.alpha) * pose_norm_diff + (1 - args.alpha) * emds_norm_diff

        # Sorting the reference tensors according to their distance
        diff_sort_idx = torch.argsort(weigthed_diff)

        # Setting the indices to extract based on the number of requested samples
        selected_refs_idx = np.linspace(1, (embd_data.shape[0] - 1), num=args.num_samples, dtype=int)

        # Saving the selected references to output
        samples_output[idx, :] = selected_refs_idx

    pd.DataFrame(samples_output.astype(np.int)).to_csv('pos_neg_samples.csv')