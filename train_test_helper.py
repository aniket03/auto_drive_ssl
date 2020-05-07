import torch
import torch.nn.functional as F

import numpy as np

from torch.nn.utils import clip_grad_norm_

from common_constants import NUM_SAMPLE_PER_SCENE
from pirl_loss import loss_pirl, get_img_pair_probs


def get_count_correct_preds(network_output, target):

    score, predicted = torch.max(network_output, 1)  # Returns max score and the index where max score was recorded
    count_correct = (target == predicted).sum().float()  # So that when accuracy is computed, it is not rounded to int

    return count_correct


def get_count_correct_preds_pretext(img_pair_probs_arr, img_mem_rep_probs_arr):
    """
    Get count of correct predictions for pre-text task
    :param img_pair_probs_arr: Prob vector of batch of images I and I_t to belong to same data distribution.
    :param img_mem_rep_probs_arr: Prob vector of batch of I and mem_bank_rep of I to belong to same data distribution
    """

    avg_probs_arr = (1/2) * (img_pair_probs_arr + img_mem_rep_probs_arr)
    count_correct = (avg_probs_arr >= 0.5).sum().float()  # So that when accuracy is computed, it is not rounded to int

    return count_correct.item()


def get_threat_score(output, target):
    """
    Get average threat score between output and target
    :param output: batch of road maps
    :param target: batch of target road maps.
    """

    # Make output to 0-1 form
    output_1 = output.round()

    # Reshape output and target
    output_1 = output_1.reshape(-1,)
    target = target.reshape(-1,)

    tp = (output_1 * target).sum().float()
    fp = (output_1 * (1-target)).sum().float()
    fn = ((1-output_1) * target).sum().float()

    return tp/(tp+fp+fn)


class SimCLRModelTrainTest():

    def __init__(self, aux_model, main_model, device, model_file_path, all_samples_mem, train_scene_indices,
                 val_scene_indices, count_negatives, temp_parameter, beta, only_train=False, threshold=1e-4):
        super(SimCLRModelTrainTest, self).__init__()
        self.aux_model = aux_model
        self.main_model = main_model
        self.device = device
        self.model_file_path = model_file_path
        self.threshold = threshold
        self.train_loss = 1e9
        self.val_loss = 1e9
        self.all_samples_mem = torch.tensor(all_samples_mem, dtype=torch.float).to(device)
        self.train_scene_indices = train_scene_indices.copy()
        self.val_scene_indices = val_scene_indices.copy()
        self.count_negatives = count_negatives
        self.temp_parameter = temp_parameter
        self.beta = beta
        self.only_train = only_train

    def get_transformed_batch(self, i_batch):
        # Rotate
        rotation_var = torch.argmax(torch.rand(3))
        if rotation_var == 0:
            # Rotate 90deg
            i_t_batch = torch.transpose(i_batch, 2, 3)
        elif rotation_var == 1:
            # Rotate 180deg
            i_t_batch = torch.flip(i_batch, [2])
        else:
            # Rotate 270deg
            i_t_batch = torch.transpose(i_batch, 2, 3)
            i_t_batch = torch.flip(i_t_batch, [3])

        # TODO: Add Gaussian noise code

        # Detach i_t_batch from computation graph of i_batch
        i_t_batch = i_t_batch.clone().detach()

        return i_t_batch

    def get_sample_indices_for_scenes(self, scene_indices):
        sample_indices = []
        for scene_index in scene_indices:
            st_sample_index = scene_index * NUM_SAMPLE_PER_SCENE
            en_sample_index = (scene_index + 1) * NUM_SAMPLE_PER_SCENE
            sample_indices += range(st_sample_index, en_sample_index)
        sample_indices = np.array(sample_indices)

        return sample_indices

    def train(self, optimizer, epoch, params_max_norm, train_data_loader, val_data_loader,
              no_train_samples, no_val_samples):
        self.aux_model.train()
        self.main_model.train()
        train_loss, correct, cnt_batches = 0, 0, 0

        for batch_idx, (data_batch, batch_scene_indices, batch_sample_indices) in enumerate(train_data_loader):

            # print ('Batch idx', batch_idx)
            # if batch_idx > 1:
            #     break

            # Set device for data_batch and batch_scene_indices
            data_batch = data_batch.to(self.device)
            batch_scene_indices = batch_scene_indices.to(self.device)
            batch_sample_indices = batch_sample_indices.to(self.device)

            # Pass data_batch through the aux_model to get latent 800x800 representation
            i_batch = self.aux_model(data_batch)
            del data_batch

            # Get transformed version of latent_representation (i_batch)
            i_t_batch = self.get_transformed_batch(i_batch)

            # Forward pass through the main network
            optimizer.zero_grad()
            vi_batch, vi_t_batch = self.main_model(i_batch, i_t_batch)
            del i_batch, i_t_batch

            # Prepare memory bank of negatives for current batch
            mn_scene_indices_all = np.array(list(set(self.train_scene_indices) - set(batch_scene_indices)))
            mn_sample_indices_all = self.get_sample_indices_for_scenes(mn_scene_indices_all)
            np.random.shuffle(mn_sample_indices_all)
            mn_indices = mn_sample_indices_all[:self.count_negatives]
            mn_arr = self.all_samples_mem[mn_indices]

            # Get memory bank representation for current batch images
            mem_rep_of_batch_imgs = self.all_samples_mem[batch_sample_indices]

            # Get prob for I, I_t to belong to same data distribution.
            img_pair_probs_arr = get_img_pair_probs(vi_batch, vi_t_batch, mn_arr, self.temp_parameter)

            # Get prob for I and mem_bank_rep of I to belong to same data distribution
            img_mem_rep_probs_arr = get_img_pair_probs(vi_batch, mem_rep_of_batch_imgs, mn_arr, self.temp_parameter)
            del mn_arr

            # Compute loss => back-prop gradients => Update weights
            loss = loss_pirl(img_pair_probs_arr, img_mem_rep_probs_arr)
            loss.backward()

            clip_grad_norm_(self.main_model.parameters(), params_max_norm)
            clip_grad_norm_(self.aux_model.parameters(), params_max_norm)
            optimizer.step()

            # Update running loss and no of pseudo correct predictions for epoch
            correct += get_count_correct_preds_pretext(img_pair_probs_arr, img_mem_rep_probs_arr)
            train_loss += loss.item()
            cnt_batches += 1

            # Update memory bank representation for images from current batch
            all_samples_mem_new = self.all_samples_mem.clone().detach()
            all_samples_mem_new[batch_sample_indices] = (self.beta * all_samples_mem_new[batch_sample_indices]) + \
                                                        ((1 - self.beta) * vi_batch)
            self.all_samples_mem = all_samples_mem_new.clone().detach()

            del batch_scene_indices, batch_sample_indices
            del vi_batch, vi_t_batch, mem_rep_of_batch_imgs
            del img_mem_rep_probs_arr, img_pair_probs_arr, all_samples_mem_new

        train_loss /= cnt_batches

        if epoch % 10 == 0:
            torch.save(self.aux_model.state_dict(), self.model_file_path + '_aux_epoch_{}'.format(epoch))
            torch.save(self.main_model.state_dict(), self.model_file_path + '_main_epoch_{}'.format(epoch))

        if self.only_train is False:
            val_loss, val_acc = self.test(epoch, val_data_loader, no_val_samples)

            if val_loss < self.val_loss - self.threshold:
                self.val_loss = val_loss
                torch.save(self.aux_model.state_dict(), self.model_file_path + '_aux')
                torch.save(self.main_model.state_dict(), self.model_file_path + '_main')

        else:
            val_loss, val_acc = 0.0, 0.0

        train_acc = correct / no_train_samples

        print('\nAfter epoch {} - Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, train_loss, correct, no_train_samples, 100. * correct / no_train_samples))

        return train_loss, train_acc, val_loss, val_acc

    def test(self, epoch, test_data_loader, no_test_samples):

        self.aux_model.eval()
        self.main_model.eval()
        test_loss, correct, cnt_batches = 0, 0, 0

        for batch_idx, (data_batch, batch_scene_indices, batch_sample_indices) in enumerate(test_data_loader):

            # Set device for data_batch and batch_scene_indices
            data_batch = data_batch.to(self.device)
            batch_scene_indices = batch_scene_indices.to(self.device)
            batch_sample_indices = batch_sample_indices.to(self.device)

            # Pass data_batch through the aux_model to get latent 800x800 representation
            i_batch = self.aux_model(data_batch)
            del data_batch

            # Get transformed version of latent_representation (i_batch)
            i_t_batch = self.get_transformed_batch(i_batch)

            # Forward pass through the main network
            vi_batch, vi_t_batch = self.main_model(i_batch, i_t_batch)
            del i_batch, i_t_batch

            # Prepare memory bank of negatives for current batch
            mn_scene_indices_all = np.array(list(set(self.val_scene_indices) - set(batch_scene_indices)))
            mn_sample_indices_all = self.get_sample_indices_for_scenes(mn_scene_indices_all)
            np.random.shuffle(mn_sample_indices_all)
            mn_indices = mn_sample_indices_all[:self.count_negatives]
            mn_arr = self.all_samples_mem[mn_indices]

            # Get memory bank representation for current batch images
            mem_rep_of_batch_imgs = self.all_samples_mem[batch_sample_indices]

            # Get prob for I, I_t to belong to same data distribution.
            img_pair_probs_arr = get_img_pair_probs(vi_batch, vi_t_batch, mn_arr, self.temp_parameter)

            # Get prob for I and mem_bank_rep of I to belong to same data distribution
            img_mem_rep_probs_arr = get_img_pair_probs(vi_batch, mem_rep_of_batch_imgs, mn_arr, self.temp_parameter)
            del mn_arr

            # Compute loss
            loss = loss_pirl(img_pair_probs_arr, img_mem_rep_probs_arr)

            # Update running loss and no of pseudo correct predictions for epoch
            correct += get_count_correct_preds_pretext(img_pair_probs_arr, img_mem_rep_probs_arr)
            test_loss += loss.item()
            cnt_batches += 1

            # Update memory bank representation for images from current batch
            all_samples_mem_new = self.all_samples_mem.clone().detach()
            all_samples_mem_new[batch_sample_indices] = (self.beta * all_samples_mem_new[batch_sample_indices]) + \
                                                        ((1 - self.beta) * vi_batch)
            self.all_samples_mem = all_samples_mem_new.clone().detach()

            del batch_scene_indices, batch_sample_indices
            del vi_batch, vi_t_batch, mem_rep_of_batch_imgs
            del img_mem_rep_probs_arr, img_pair_probs_arr, all_samples_mem_new

        test_loss /= cnt_batches
        test_acc = correct / no_test_samples
        print('\nAfter epoch {} - Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, test_loss, correct, no_test_samples, 100. * correct / no_test_samples))

        return  test_loss, test_acc


class FCNModelTrainTest():

    def __init__(self, aux_model, main_model, device, model_file_path, threshold=1e-4):
        super(FCNModelTrainTest, self).__init__()
        self.aux_model = aux_model
        self.main_model = main_model
        self.device = device
        self.model_file_path = model_file_path
        self.threshold = threshold
        self.train_loss = 1e9
        self.val_loss = 1e9

    def get_transformed_batch(self, data, target):

        flip_var = torch.argmax(torch.rand(3))
        if flip_var == 0:
            # Do nothing
            pass
        elif flip_var == 1:
            # Horizontal flip input, thus vertical flip target
            data = torch.flip(data, [3])
            target = torch.flip(target, [2])
        else:
            # Vertical flip input, thus horizontal flip target
            data = torch.flip(data, [2])
            target = torch.flip(target, [3])

        return data, target

    def train(self, optimizer, epoch, params_max_norm, train_data_loader, val_data_loader,
              no_train_samples, no_val_samples):
        self.aux_model.train()
        self.main_model.train()
        train_loss, cnt_batches = 0, 0
        correct = 0

        for batch_idx, (data, target) in enumerate(train_data_loader):

            data, target = data.to(self.device), target.to(device=self.device, dtype=torch.float)

            target = target.reshape(-1, 1, 800, 800)

            optimizer.zero_grad()
            pseudo_input = self.aux_model(data)

            # tr_pseudo_input, target  = self.get_transformed_batch(pseudo_input, target)
            tr_pseudo_input = pseudo_input

            output = self.main_model(tr_pseudo_input)

            loss = F.binary_cross_entropy(output, target)
            loss.backward()

            clip_grad_norm_(self.aux_model.parameters(), params_max_norm)
            clip_grad_norm_(self.main_model.parameters(), params_max_norm)
            optimizer.step()

            # correct += get_count_correct_preds(output, target)
            train_loss += loss.item()
            cnt_batches += 1

            del data, target, output

        train_loss /= cnt_batches
        val_loss, val_acc = self.test(epoch, val_data_loader, no_val_samples)

        if epoch % 10 == 0:
            torch.save(self.aux_model.state_dict(), self.model_file_path + '_aux_fcn_epoch_{}'.format(epoch))
            torch.save(self.main_model.state_dict(), self.model_file_path + '_main_fcn_epoch_{}'.format(epoch))

        if val_loss < self.val_loss - self.threshold:
            self.val_loss = val_loss
            torch.save(self.aux_model.state_dict(), self.model_file_path+'_aux_fcn')
            torch.save(self.main_model.state_dict(), self.model_file_path+'_main_fcn')

        train_acc = correct / no_train_samples

        print('\nAfter epoch {} - Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, train_loss, correct, no_train_samples, 100. * correct / no_train_samples))

        return train_loss, train_acc, val_loss, val_acc

    def test(self, epoch, test_data_loader, no_test_samples):
        self.aux_model.eval()
        self.main_model.eval()
        test_loss = 0
        test_acc = 0
        cnt_batches = 0

        for batch_idx, (data, target) in enumerate(test_data_loader):

            data, target = data.to(self.device), target.to(self.device, dtype=torch.float)
            target = target.reshape(-1, 1, 800, 800)

            pseudo_input = self.aux_model(data)
            output = self.main_model(pseudo_input)

            test_loss += F.binary_cross_entropy(output, target).item()  # sum up batch loss
            test_acc += get_threat_score(output, target).item()

            # correct += get_count_correct_preds(output, target)
            cnt_batches += 1

            del data, target, output

        test_loss /= cnt_batches
        test_acc /= cnt_batches
        print('\nAfter epoch {} - Test set: Average loss: {:.4f}, Accuracy: ({:.2f}%)\n'.format(
            epoch, test_loss, test_acc))

        return  test_loss, test_acc



class ModelTrainTest():

    def __init__(self, network, device, model_file_path, threshold=1e-4):
        super(ModelTrainTest, self).__init__()
        self.network = network
        self.device = device
        self.model_file_path = model_file_path
        self.threshold = threshold
        self.train_loss = 1e9
        self.val_loss = 1e9

    def train(self, optimizer, epoch, params_max_norm, train_data_loader, val_data_loader,
              no_train_samples, no_val_samples):
        self.network.train()
        train_loss, correct, cnt_batches = 0, 0, 0

        for batch_idx, (data, target) in enumerate(train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.network(data)

            loss = F.nll_loss(output, target)
            loss.backward()

            clip_grad_norm_(self.network.parameters(), params_max_norm)
            optimizer.step()

            correct += get_count_correct_preds(output, target)
            train_loss += loss.item()
            cnt_batches += 1

            del data, target, output

        train_loss /= cnt_batches
        val_loss, val_acc = self.test(epoch, val_data_loader, no_val_samples)

        if val_loss < self.val_loss - self.threshold:
            self.val_loss = val_loss
            torch.save(self.network.state_dict(), self.model_file_path)

        train_acc = correct / no_train_samples

        print('\nAfter epoch {} - Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, train_loss, correct, no_train_samples, 100. * correct / no_train_samples))

        return train_loss, train_acc, val_loss, val_acc

    def test(self, epoch, test_data_loader, no_test_samples):
        self.network.eval()
        test_loss = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(test_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss

            correct += get_count_correct_preds(output, target)

            del data, target, output

        test_loss /= no_test_samples
        test_acc = correct / no_test_samples
        print('\nAfter epoch {} - Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, test_loss, correct, no_test_samples, 100. * correct / no_test_samples))

        return  test_loss, test_acc

if __name__ == '__main__':
    img_pair_probs_arr = torch.randn((256,))
    img_mem_rep_probs_arr = torch.randn((256,))
    print (get_count_correct_preds_pretext(img_pair_probs_arr, img_mem_rep_probs_arr))

    output = torch.rand(2, 1, 800, 800)
    target = torch.rand(2, 1, 800, 800)
    target = target.round()
    x = get_threat_score(output, target)
    print (x)
