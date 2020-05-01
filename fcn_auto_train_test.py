import os
import argparse

import torch

import numpy as np

from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from common_constants import PAR_WEIGHTS_DIR, PAR_ACTIVATIONS_DIR
from dataset_helpers import def_train_transform, brightness_jitter_transform
from experiment_logger import log_experiment
from get_dataset import LabeledDataset
from models import CombineAndUpSample, fcn_resnet, simclr_resnet
from network_helpers import copy_weights_between_models, test_copy_weights
from random_seed_setter import set_random_generators_seed
from train_test_helper import FCNModelTrainTest

if __name__ == '__main__':

    # Training arguments
    parser = argparse.ArgumentParser(description='Self driving train test script for Semantic segmentation task')
    parser.add_argument('--model-type', type=str, default='res34', help='The network architecture '
                                                                        'to employ as backbone')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 150)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay constant (default: 5e-4)')
    parser.add_argument('--tmax-for-cos-decay', type=int, default=50)
    parser.add_argument('--early-stop-patience', type=int, default=7)
    parser.add_argument('--warm-start', type=bool, default=False)
    parser.add_argument('--only-train', type=bool, default=False,
                        help='If true utilize the annotated dataset for training.')
    parser.add_argument('--ssl-trained-aux-file', type=str, default='')
    parser.add_argument('--ssl-trained-main-file', type=str, default='')
    parser.add_argument('--experiment-name', type=str, default='e1_sem_seg_auto_')
    args = parser.parse_args()

    # Set random number generation seed for all packages that generate random numbers
    set_random_generators_seed()

    # Identify device for holding tensors and carrying out computations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define train_set, val_set objects
    base_images_dir = '../data'
    annotation_file = os.path.join(base_images_dir, 'annotation.csv')
    scene_indices = np.arange(111, 134)
    np.random.shuffle(scene_indices)
    train_scene_indices = scene_indices[:19]
    val_scene_indices = scene_indices[19:]
    train_set = LabeledDataset(base_images_dir, annotation_file, train_scene_indices,
                               transform=brightness_jitter_transform)
    val_set = LabeledDataset(base_images_dir, annotation_file, val_scene_indices,
                             transform=def_train_transform)

    # Define train and validation data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True,
                                             num_workers=8)

    # Print sample batches that would be returned by the train_data_loader
    dataiter = iter(train_loader)
    X, y = dataiter.__next__()
    print (len(train_set))
    print (X.size())
    print (y.size())

    # Train required model using data loaders defined above
    epochs = args.epochs
    weight_decay_const = args.weight_decay

    # Define model(s) to train
    aux_model = CombineAndUpSample(n_feature=64)
    main_model = fcn_resnet(args.model_type, 1)

    # Set device on which training is done.
    aux_model.to(device)
    main_model.to(device)

    # Inherit weights for aux model from pre-trained SSL aux model
    aux_file_path = os.path.join(PAR_WEIGHTS_DIR, args.ssl_trained_aux_file)
    aux_model.load_state_dict(torch.load(aux_file_path, map_location=device))

    # Inherit weights for main model from pre-trained SSL main model
    simclr_model = simclr_resnet(args.model_type, non_linear_head=False)
    simclr_model_file_path = os.path.join(PAR_WEIGHTS_DIR, args.ssl_trained_main_file)
    simclr_model.load_state_dict(torch.load(simclr_model_file_path, map_location=device))
    simclr_model.to(device)
    weight_copy_success = copy_weights_between_models(simclr_model, main_model)
    del simclr_model

    # Freeze all layers in the aux model and main model's resnet module
    for name, param in aux_model.named_parameters():
        param.requires_grad = False
    for name, param in main_model.named_parameters():
        if name[:7] == 'resnet_':
            param.requires_grad = False

    unblock_levels = [0, 1, 2, 3]  # 0=> block all, 1=> block till residual block 3, 2=> block till residual block 2
                                   # 3 means unblock all
    for unblock_level in unblock_levels:

        if unblock_level == 0:
            lr = 0.5

        if unblock_level == 1:
            lr = 0.1
            for name, param in main_model.named_parameters():
                if name[:15] == 'resnet_module.7':
                    param.requires_grad = True

        if unblock_level == 2:
            lr = 0.05
            for name, param in main_model.named_parameters():
                if name[:15] == 'resnet_module.6' or name[:15] == 'resnet_module.7':
                    param.requires_grad = True

        else:
            lr = 0.01
            for name, param in aux_model.named_parameters():
                param.requires_grad = True
            for name, param in main_model.named_parameters():
                param.requires_grad = True

        # # To test what is trainable status of each layer
        # for name, param in aux_model.named_parameters():
        #     print (name, param.requires_grad)
        # for name, param in main_model.named_parameters():
        #     print (name, param.requires_grad)

        # Define the file_path where trained model will be saved
        model_file_path = os.path.join(PAR_WEIGHTS_DIR,
                                       args.experiment_name + '_ub_level_{}'.format(unblock_level))

        # Start training
        model_train_test_obj = FCNModelTrainTest(
            aux_model, main_model, device, model_file_path
        )
        params = list(aux_model.parameters()) + list(main_model.parameters())
        sgd_optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay_const)
        scheduler = CosineAnnealingLR(sgd_optimizer, args.tmax_for_cos_decay, eta_min=1e-4, last_epoch=-1)
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        init_val_loss = 1e9
        val_loss_threshold = 1e-4
        early_stop_cnt = 0
        for epoch_no in range(1, epochs + 1):
            train_loss, train_acc, val_loss, val_acc = model_train_test_obj.train(
                sgd_optimizer, epoch_no, params_max_norm=4,
                train_data_loader=train_loader, val_data_loader=val_loader,
                no_train_samples=len(train_set), no_val_samples=len(val_set)
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            scheduler.step()

            if val_loss < init_val_loss - val_loss_threshold:
                init_val_loss = val_loss
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
                if early_stop_cnt >= args.early_stop_patience:
                    print ('Val loss not decreasing so doing early stop')
                    break

        # Log train-test results
        n_epochs_ran = len(train_losses)
        log_experiment(args.experiment_name + '_ub_level_{}'.format(unblock_level),
                       n_epochs_ran, train_losses, val_losses, train_accs, val_accs)
