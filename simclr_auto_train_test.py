import os
import argparse

import torch

import numpy as np

from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from common_constants import PAR_WEIGHTS_DIR
from dataset_helpers import def_train_transform
from experiment_logger import log_experiment
from get_dataset import UnlabeledDataset
from models import simclr_resnet, CombineAndUpSample
from random_seed_setter import set_random_generators_seed
from train_test_helper import SimCLRModelTrainTest


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':

    # Training arguments
    parser = argparse.ArgumentParser(description='Self driving train test script for SSL task')
    parser.add_argument('--model-type', type=str, default='res50', help='The network architecture to employ as backbone')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 150)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay constant (default: 5e-4)')
    parser.add_argument('--tmax-for-cos-decay', type=int, default=50)
    # parser.add_argument('--warm-start', type=bool, default=False)
    parser.add_argument('--count-negatives', type=int, default=6400,
                        help='No of samples in memory bank of negatives')
    parser.add_argument('--beta', type=float, default=0.5, help='Exponential running average constant'
                                                                'in memory bank update')
    parser.add_argument('--only-train', type=bool, default=False,
                        help='If true utilize the entire unannotated STL10 dataset for training.')
    parser.add_argument('--non-linear-head', type=bool, default=False,
                        help='If true apply non-linearity to the output of function heads '
                             'applied to resnet image representations')
    parser.add_argument('--temp-parameter', type=float, default=0.07, help='Temperature parameter in NCE probability')
    # parser.add_argument('--cont-epoch', type=int, default=1, help='Epoch to start the training from, helpful when using'
    #                                                               'warm start')
    parser.add_argument('--experiment-name', type=str, default='e1_pirl_auto_')
    args = parser.parse_args()

    # Set random number generation seed for all packages that generate random numbers
    set_random_generators_seed()

    # Identify device for holding tensors and carrying out computations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the file_path where trained model will be saved
    model_file_path = os.path.join(PAR_WEIGHTS_DIR, args.experiment_name)

    # Define train_set, val_set objects
    base_images_dir = '../data'
    scene_indices = np.arange(0, 111)
    train_set = UnlabeledDataset(base_images_dir, scene_indices, first_dim='sample',
                                 transform=def_train_transform)

    # Define train and validation data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=8)

    # Print sample batches that would be returned by the train_data_loader
    dataiter = iter(train_loader)
    X, y = dataiter.__next__()
    print (len(train_set))
    print (X[0].size())
    print (X[1].size())
    print (y.size())

    # Train required model using data loaders defined above
    epochs = args.epochs
    lr = args.lr
    weight_decay_const = args.weight_decay

    # Define model(s) to train
    aux_model = CombineAndUpSample(n_feature=64)
    main_model = simclr_resnet(args.model_type, args.non_linear_head)

    # Set device on which training is done. Plus optimizer to use.
    aux_model.to(device)
    main_model.to(device)
    params = list(aux_model.parameters()) + list(main_model.parameters())
    sgd_optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay_const)
    scheduler = CosineAnnealingLR(sgd_optimizer, args.tmax_for_cos_decay, eta_min=1e-4, last_epoch=-1)

    # # Initialize model weights with a previously trained model if using warm start
    # if args.warm_start and os.path.exists(model_file_path):
    #     model_to_train.load_state_dict(torch.load(model_file_path, map_location=device))

    # Start training
    all_samples_mem = np.random.randn(len(train_set), 128)
    val_indices = []
    model_train_test_obj = SimCLRModelTrainTest(
        aux_model, main_model, device, model_file_path, all_samples_mem, scene_indices, val_indices,
        args.count_negatives, args.temp_parameter, args.beta, args.only_train
    )
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch_no in range(1, epochs + 1):
        train_loss, train_acc, val_loss, val_acc = model_train_test_obj.train(
            sgd_optimizer, epoch_no, params_max_norm=4,
            train_data_loader=train_loader, val_data_loader=None,
            no_train_samples=len(scene_indices), no_val_samples=len(val_indices)
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step()

    # Log train-test results
    log_experiment(args.experiment_name, args.epochs, train_losses, val_losses, train_accs, val_accs)
