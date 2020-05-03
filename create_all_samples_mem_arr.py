import  numpy as np
import torch

from common_constants import PAR_WEIGHTS_DIR, PAR_ACTIVATIONS_DIR, NUM_SAMPLE_PER_SCENE
from dataset_helpers import def_train_transform
from get_dataset import UnlabeledDataset
from models import CombineAndUpSample, simclr_resnet


def get_all_samples_mem_arr(device, sample_data_loader, aux_model, main_model,
                            all_samples_mem_file):
    aux_model.eval()
    main_model.eval()

    all_samples_mem = None
    for batch_idx, (data_batch, batch_scene_indices, batch_sample_indices) in enumerate(sample_data_loader):
        # Set device for data_batch and batch_scene_indices
        data_batch = data_batch.to(device)
        batch_scene_indices = batch_scene_indices.to(device)
        batch_sample_indices = batch_sample_indices.to(device)

        with torch.no_grad():
            # Pass data_batch through the aux_model to get latent 800x800 representation
            i_batch = aux_model(data_batch)
            del data_batch

            # Forward pass through the main network
            vi_batch, vi_t_batch = main_model(i_batch, i_batch)
            del i_batch, vi_t_batch

            # Convert vi_batch to a numpy array
            vi_batch = vi_batch.numpy()
            if all_samples_mem is None:
                all_samples_mem = vi_batch
            else:
                all_samples_mem = np.concatenate([all_samples_mem, vi_batch], axis=0)

            del batch_scene_indices, batch_sample_indices
            del vi_batch

    np.save(all_samples_mem_file, all_samples_mem)


if __name__ == '__main__':

    # Identify device for holding tensors and carrying out computations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define model file_paths
    aux_model_file = PAR_WEIGHTS_DIR + '/e1_simclr_auto_aux_epoch_90'
    main_model_file = PAR_WEIGHTS_DIR + '/e1_simclr_auto_main_epoch_90'
    all_samples_mem_file = PAR_ACTIVATIONS_DIR + '/e1_simclr_auto_activ_epoch_90.npy'

    # Get the initial random weight models
    aux_model = CombineAndUpSample(n_feature=64)
    main_model = simclr_resnet('res34', non_linear_head=False)

    # Initialize model weights with a previously trained model
    aux_model.load_state_dict(torch.load(aux_model_file, map_location=device))
    main_model.load_state_dict(torch.load(main_model_file, map_location=device))

    # Get data loader
    base_images_dir = '../data'
    scene_indices = np.arange(0, 111)
    sample_data_set = UnlabeledDataset(base_images_dir, scene_indices, first_dim='sample',
                                       transform=def_train_transform)
    sample_data_loader = torch.utils.data.DataLoader(sample_data_set, batch_size=scene_indices.size,
                                                     num_workers=32)
    # Get the memory bank representation required
    get_all_samples_mem_arr(device, sample_data_loader, aux_model, main_model,
                            all_samples_mem_file)









