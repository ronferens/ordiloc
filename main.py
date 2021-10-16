"""
Entry point training and testing TransPoseNet
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from models.pose_losses import CameraPoseLoss, CameraPoseOrdinalLoss
from models.pose_regressors import get_model
from os.path import join
from sklearn.metrics import confusion_matrix
from util import plotutils
from util import visdomutils

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_name",
                            help="name of model to create (e.g. posenet, transposenet")
    arg_parser.add_argument("mode", help="train or eval")
    arg_parser.add_argument("backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")
    arg_parser.add_argument("--train_labels_file", help="used for loading the clusters' centroids")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {} with {}".format(args.model_name, args.mode))
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open("config.json", "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.model_name]
    general_params = config['general']
    config = {**model_params, **general_params}
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Init Visdom context
    visdom_active = config.get('device_id')
    if visdom_active:
        plotter = visdomutils.VisdomLinePlotter(env_name=utils.get_stamp_from_log())

    # Create the model
    model = get_model(args.model_name, args.backbone_path, config).to(device)

    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    if args.mode == 'train':
        # Set to train mode
        model.train()

        cent_pos, cent_orient = utils.load_clusters_centroids(args.labels_file, device)
        model.set_centroids(cent_pos, cent_orient)

        # Freeze parts of the model if indicated
        freeze = config.get("freeze")
        freeze_exclude_phrase = config.get("freeze_exclude_phrase")
        if isinstance(freeze_exclude_phrase, str):
            freeze_exclude_phrase = [freeze_exclude_phrase]
        if freeze:
            for name, parameter in model.named_parameters():
                freeze_param = True
                for phrase in freeze_exclude_phrase:
                    if phrase in name:
                        freeze_param = False
                        break
                if freeze_param:
                        parameter.requires_grad_(False)

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)
        pose_ordi_loss = CameraPoseOrdinalLoss(config).to(device)

        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        # Set the dataset and data loader
        no_augment = config.get("no_augment")
        if no_augment:
            transform = utils.test_transforms.get('baseline')
        else:
            transform = utils.train_transforms.get('baseline')

        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform)
        loader_params = {'batch_size': config.get('batch_size'),
                         'shuffle': True,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")
        ordi_loss_weight = config.get("ordi_loss_weight")

        #  Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            running_loss_pose = 0.0
            running_loss_pose_ordi = 0.0
            n_samples = 0

            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)

                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                gt_pose_cls = minibatch.get('pose_cls').to(dtype=torch.float32)
                gt_orient_cls = minibatch.get('orient_cls').to(dtype=torch.float32)
                batch_size = gt_pose.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                # Zero the gradients
                optim.zero_grad()

                # Forward pass to estimate the pose
                res = model(minibatch)
                est_pose = res.get('pose')
                est_pose_cls = res.get('pose_cls')
                est_orient_cls = res.get('orient_cls')

                # Calculating the losses
                pose_loss_val = pose_loss(est_pose, gt_pose)
                pose_ordi_loss_val = ordi_loss_weight * pose_ordi_loss([est_pose_cls, est_orient_cls],
                                                                       [gt_pose_cls, gt_orient_cls])
                criterion = pose_loss_val + pose_ordi_loss_val
                # criterion = pose_loss(est_pose, gt_pose) + ordi_loss_weight * pose_ordi_loss(est_pose_cls, gt_pose_cls)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                running_loss_pose += pose_loss_val.item()
                running_loss_pose_ordi += pose_ordi_loss_val.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(est_pose, gt_pose)
                    logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                                 "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_samples),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item()))

                    pose_class_err, orient_class_err = utils.pose_class_err([est_pose_cls.detach(),
                                                                             est_orient_cls.detach()],
                                                                            [gt_pose_cls.detach(),
                                                                             gt_orient_cls.detach()])
                    logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                                 "Pose class error: Position={:.2f}%, Orientation={:.2f}%".format(
                        batch_idx + 1, epoch + 1, (running_loss / n_samples),
                        100. * torch.sum(pose_class_err).item() / pose_class_err.shape[0],
                        100. * torch.sum(orient_class_err).item() / pose_class_err.shape[0]))

            if visdom_active:
                plotter.plot('pose_loss', 'train', 'Pose Loss', epoch, pose_loss_val.item())
                plotter.plot('pose_ordi_loss', 'train', 'Ordinal Classification Loss', epoch, pose_ordi_loss_val.item())
                plotter.plot('running_loss', 'train', 'Running Loss', epoch, running_loss)

            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        logging.info('Final mode: ' + checkpoint_prefix + '_final.pth')
        torch.save(model.state_dict(), checkpoint_prefix + '_final.pth')

        # Plot the loss function
        loss_fig_path = checkpoint_prefix + "_loss_fig.png"
        utils.plot_loss_func(sample_count, loss_vals, loss_fig_path)

    else:  # Test
        # Set to eval mode
        model.eval()

        if args.train_labels_file is None:
            raise 'In test mode you must supply the \'train_dataset_path\' argument'
        else:
            cent_pos, cent_orient = utils.load_clusters_centroids(args.train_labels_file, device)
            model.set_centroids(cent_pos, cent_orient)

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        stats = np.zeros((len(dataloader.dataset), 5))

        gt = {'Position': [], 'Orientation': []}
        preds = {'Position': [], 'Orientation': []}

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)

                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                gt_pose_cls = minibatch.get('pose_cls').to(dtype=torch.float32)
                gt_orient_cls = minibatch.get('orient_cls').to(dtype=torch.float32)

                # Forward pass to predict the pose
                tic = time.time()
                res = model(minibatch)
                toc = time.time()

                est_pose = res.get('pose')
                est_pose_cls = res.get('pose_cls')
                est_orient_cls = res.get('orient_cls')

                # Evaluate pose error
                posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

                # Collect statistics
                stats[i, 0] = posit_err.item()
                stats[i, 1] = orient_err.item()
                stats[i, 2] = (toc - tic) * 1000

                logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    stats[i, 0],  stats[i, 1],  stats[i, 2]))

                # Saving the predictions for the final confusion matrix
                preds['Position'].append(utils.convert_pred_to_label(est_pose_cls.detach()).data.cpu().numpy())
                preds['Orientation'].append(utils.convert_pred_to_label(est_orient_cls.detach()).data.cpu().numpy())
                gt['Position'].append(gt_pose_cls.data.cpu().numpy())
                gt['Orientation'].append(gt_orient_cls.data.cpu().numpy())

                # Evaluate pose's ordinal classification error
                pose_class_err, orient_class_err = utils.pose_class_err([est_pose_cls.detach(),
                                                                         est_orient_cls.detach()],
                                                                        [gt_pose_cls.detach(),
                                                                         gt_orient_cls.detach()])
                logging.info("Pose class error: Position={}, Orientation={}".format(pose_class_err.item(),
                                                                                    orient_class_err.item()))
                # Collect statistics
                stats[i, 3] = pose_class_err.item()
                stats[i, 4] = orient_class_err.item()

        # Record overall statistics
        logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
        logging.info("\tMedian pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]),
                                                                          np.nanmedian(stats[:, 1])))
        logging.info("\tPose class error: Position={:.2f}%, Orientation={:.2f}%".format(
            100. * np.sum(stats[:, 3])/stats.shape[0],
            100. * np.sum(stats[:, 4]/stats.shape[0])))
        logging.info("\tMean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))

        for cluster_type in ['Position', 'Orientation']:
            conf_matrix = confusion_matrix(y_true=np.array(preds[cluster_type]), y_pred=np.array(gt[cluster_type]))
            target_names = ['Segment #{}'.format(i) for i in range(np.max(gt[cluster_type]).astype(np.int32))]
            plotutils.plot_confusion_matrix(conf_matrix, target_names, title='Confusion matrix - ' + cluster_type,
                                            cmap=None, normalize=True)
