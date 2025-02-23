import os
import argparse

import torch

import numpy as np

from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet34

from common_constants import PAR_WEIGHTS_DIR
from dataset_helpers import def_train_transform, brightness_jitter_transform
from experiment_logger import log_experiment
from get_dataset import LabeledDataset
from models import CombineAndUpSample, fcn_resnet, fcn_resnet8s
from random_seed_setter import set_random_generators_seed
from train_test_helper import FCNModelTrainTest


def copy_weights_between_models_local(m1, m2):
    """
    Copy weights for layers common between m1 and m2.
    From m1 => m2
    """

    # Load state dictionaries for m1 model and m2 model
    m1_state_dict = m1.state_dict()
    m2_state_dict = m2.state_dict()

    # Get m1 and m2 layer names
    m1_layer_names, m2_layer_names = [], []
    for name, param in m1_state_dict.items():
        m1_layer_names.append(name)
    for name, param in m2_state_dict.items():
        m2_layer_names.append(name)

    cnt = 0
    for ind in range(len(m1_layer_names)):
        if ind < 216:
            cnt += 1
            m2_state_dict[m2_layer_names[ind]] = m1_state_dict[m1_layer_names[ind]].data

    m2.load_state_dict(m2_state_dict)

    print ('Count of layers whose weights were copied between two models', cnt)
    return m2


def test_copy_weights_resnet_module_local(m1, m2):
    """
    Tests that weights copied from m1 into m2, are actually refected in m2
    """
    m1_state_dict = m1.state_dict()
    m2_state_dict = m2.state_dict()
    weight_copy_flag = 1

    # Get m1 and m2 layer names
    m1_layer_names, m2_layer_names = [], []
    for name, param in m1_state_dict.items():
        m1_layer_names.append(name)
    for name, param in m2_state_dict.items():
        m2_layer_names.append(name)

    # Check if copy was succesful
    for ind in range(len(m1_layer_names)):
        if ind < 216:
            if not torch.all(torch.eq(m1_state_dict[m1_layer_names[ind]].data, m2_state_dict[m2_layer_names[ind]].data)):
                weight_copy_flag = 0
                print ('Something is incorrect for layer {} and {}'.format(m1_layer_names[ind], m2_layer_names[ind]))

    if weight_copy_flag == 1:
        print ('All is well')


if __name__ == '__main__':

    # Training arguments
    parser = argparse.ArgumentParser(description='Self driving train test script for Semantic segmentation task')
    parser.add_argument('--fcn-type', type=str, default='fcn32s', help='The type of fcn network to use '
                                                                       'fcn32s or fcn8s')
    parser.add_argument('--model-type', type=str, default='res34', help='The network architecture '
                                                                        'to employ as backbone')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate at which fcn8s should be learnt from '
                                                              'scratch')
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
    if args.fcn_type == 'fcn32s':
        main_model = fcn_resnet(args.model_type, 1)
    else:  #  fcn type = fcn8s
        main_model = fcn_resnet8s(args.model_type, 1)

    # Set device on which training is done.
    aux_model.to(device)
    main_model.to(device)

    # Inherit weights for main model from ImageNet pre-trained ResNet34
    imagenet_resnet = resnet34(pretrained=True)
    imagenet_resnet.to(device)
    main_model = copy_weights_between_models_local(imagenet_resnet, main_model)
    test_copy_weights_resnet_module_local(imagenet_resnet, main_model)
    del imagenet_resnet

    # Define the file_path where trained model will be saved
    model_file_path = os.path.join(PAR_WEIGHTS_DIR, args.experiment_name)

    # Start training
    model_train_test_obj = FCNModelTrainTest(
        aux_model, main_model, device, model_file_path
    )
    params = list(aux_model.parameters()) + list(main_model.parameters())
    sgd_optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=weight_decay_const)
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
    log_experiment(args.experiment_name,
                   n_epochs_ran, train_losses, val_losses, train_accs, val_accs)
