import os
import argparse

import torch

import numpy as np


from common_constants import PAR_WEIGHTS_DIR
from dataset_helpers import def_train_transform
from get_dataset import LabeledDataset
from models import CombineAndUpSample, fcn_resnet

from random_seed_setter import set_random_generators_seed
from train_test_helper import FCNModelTrainTest

if __name__ == '__main__':

    # Training arguments
    parser = argparse.ArgumentParser(description='Self driving train test script for Semantic segmentation task')
    parser.add_argument('--model-type', type=str, default='res34', help='The network architecture '
                                                                        'to employ as backbone')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--trained-aux-file', type=str, default='')
    parser.add_argument('--trained-main-file', type=str, default='')
    parser.add_argument('--experiment-name', type=str, default='e1_sem_seg_auto_')
    args = parser.parse_args()

    # Set random number generation seed for all packages that generate random numbers
    set_random_generators_seed()

    # Identify device for holding tensors and carrying out computations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the file_path where trained model will be saved
    model_file_path = os.path.join(PAR_WEIGHTS_DIR, args.experiment_name)

    # Define train_set, val_set objects
    base_images_dir = '../data'
    annotation_file = os.path.join(base_images_dir, 'annotation.csv')
    scene_indices = np.arange(111, 134)
    np.random.shuffle(scene_indices)
    train_scene_indices = scene_indices[:19]
    val_scene_indices = scene_indices[19:]
    val_set = LabeledDataset(base_images_dir, annotation_file, val_scene_indices,
                             transform=def_train_transform)

    # Define train and validation data loaders
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True,
                                             num_workers=8)

    # Print sample batches that would be returned by the train_data_loader
    dataiter = iter(val_loader)
    X, y = dataiter.__next__()
    print (len(val_set))
    print (X.size())
    print (y.size())

    # Define model(s) to test
    aux_model = CombineAndUpSample(n_feature=64)
    main_model = fcn_resnet(args.model_type, 1)

    # Set device on which training is done. Plus optimizer to use.
    aux_model.to(device)
    main_model.to(device)

    # Inherit weights for aux model from trained aux model
    aux_file_path = os.path.join(PAR_WEIGHTS_DIR, args.trained_aux_file)
    aux_model.load_state_dict(torch.load(aux_file_path, map_location=device))

    # Inherit weights for main model from trained main model
    main_model_file_path = os.path.join(PAR_WEIGHTS_DIR, args.trained_main_file)
    main_model.load_state_dict(torch.load(main_model_file_path, map_location=device))

    # Start eval
    model_train_test_obj = FCNModelTrainTest(
        aux_model, main_model, device, model_file_path
    )
    val_losses, val_accs = [], []
    val_loss, val_acc = model_train_test_obj.test(
        1, test_data_loader=val_loader,
        no_test_samples=len(val_set)
    )
    print (val_loss, val_acc)
    #     train_losses.append(train_loss)
    #     val_losses.append(val_loss)
    #     train_accs.append(train_acc)
    #     val_accs.append(val_acc)
    #     scheduler.step()
    #
    # # Log train-test results
    # log_experiment(args.experiment_name, args.epochs, train_losses, val_losses, train_accs, val_accs)
