import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50


class CombineAndUpSample(nn.Module):
    def __init__(self, n_feature):
        super(CombineAndUpSample, self).__init__()
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_feature, kernel_size=5)
        self.conv2 = nn.Conv2d(n_feature, 3, kernel_size=5)

    def forward(self, x, verbose=False):

        patch_batches = [x[:, patch_ind, :, :, :] for patch_ind in range(6)]

        for ind, patch_batch in enumerate(patch_batches):
            x = F.relu(self.conv1(patch_batch))
            x = F.relu(self.conv2(x))
            patch_batches[ind] = x

        fl_patch = patch_batches[0]
        f_patch = patch_batches[1]
        fr_patch = patch_batches[2]

        bl_patch = patch_batches[3]
        b_patch = patch_batches[4]
        br_patch = patch_batches[5]

        f_block = torch.cat((fl_patch, f_patch, fr_patch), 2)
        b_block = torch.cat((br_patch, b_patch, bl_patch), 2)
        b_block = torch.flip(b_block, [2,3])

        c_block = torch.cat((f_block, b_block))

        x = F.interpolate(c_block, size=[800,800])

        return x


class ClassificationResNet(nn.Module):

    def __init__(self, resnet_module, num_classes):
        super(ClassificationResNet, self).__init__()
        self.resnet_module = resnet_module
        self.fc = nn.Linear(512, num_classes)

    def forward(self, input_batch):

        # Data returned by data loaders is of the shape (batch_size, no_channels, h_patch, w_patch)
        resnet_feat_vectors = self.resnet_module(input_batch)
        final_feat_vectors = torch.flatten(resnet_feat_vectors, 1)
        x = F.log_softmax(self.fc(final_feat_vectors))

        return x


def get_base_resnet_module(model_type):
    """
    Returns the backbone network for required resnet architecture, specified as model_type
    :param model_type: Can be either of {res18, res34, res50}
    """

    if model_type == 'res18':
        original_model = resnet18(pretrained=False)
    elif model_type == 'res34':
        original_model = resnet34(pretrained=False)
    else:
        original_model = resnet50(pretrained=False)
    base_resnet_module = nn.Sequential(*list(original_model.children())[:-1])

    return base_resnet_module


def classifier_resnet(model_type, num_classes):
    """
    Returns a classification network with backbone belonging to the family of ResNets
    :param model_type: Specifies which resnet network to employ. Can be one of {res18, res34, res50}
    :param num_classes: The number of classes that the final network classifies it inputs into.
    """

    base_resnet_module = get_base_resnet_module(model_type)

    return ClassificationResNet(base_resnet_module, num_classes)


class SimCLRResnet(nn.Module):
    def __init__(self, resnet_module, non_linear_head=False):
        super(SimCLRResnet, self).__init__()
        self.resnet_module = resnet_module
        self.lin_project_1 = nn.Linear(512, 128)
        if non_linear_head:
            self.lin_project_2 = nn.Linear(128, 128)  # Will only be used if non_linear_head is True
        self.non_linear_head = non_linear_head

    def forward(self, i_batch, i_t_batch):
        """
        :param i_batch: Batch of images
        :param i_t_patches_batch: Batch of transformed images
        """

        # Run I and I_t through resnet
        vi_batch = self.resnet_module(i_batch)
        vi_batch = torch.flatten(vi_batch, 1)
        vi_t_batch = self.resnet_module(i_t_batch)
        vi_t_batch = torch.flatten(vi_t_batch, 1)

        # Run resnet features for I and I_t via lin_project_1 layer
        vi_batch = self.lin_project_1(vi_batch)
        vi_t_batch = self.lin_project_1(vi_t_batch)

        # Run final feature vectors obtained for I and I_t through non-linearity (if specified)
        if self.non_linear_head:
            vi_batch = self.lin_project_2(F.relu(vi_batch))
            vi_t_batch = self.lin_project_2(F.relu(vi_t_batch))

        return vi_batch, vi_t_batch


def simclr_resnet(model_type, non_linear_head=False):
    """
    Returns a network which supports Pre-text invariant representation learning
    with backbone belonging to the family of ResNets
    :param model_type: Specifies which resnet network to employ. Can be one of {res18, res34, res50}
    :param non_linear_head: If true apply non-linearity to the output of function heads
    applied to resnet image representations
    """

    base_resnet_module = get_base_resnet_module(model_type)

    return SimCLRResnet(base_resnet_module, non_linear_head)


if __name__ == '__main__':

    # Test Combine and up sample
    cm = CombineAndUpSample(64)
    patches_batch = torch.randn(32, 6, 3, 64, 64)
    result = cm.forward(patches_batch)
    print (result.shape)

    # Test SimCLR model
    sr = simclr_resnet('res18', non_linear_head=False)  # non_linear_head can be True or False either.
    image_batch = torch.randn(32, 3, 64, 64)
    tr_img_batch = torch.randn(32, 3, 64, 64)

    result1, result2 = sr.forward(image_batch, tr_img_batch)

    print (result1.size())
    print (result2.size())
