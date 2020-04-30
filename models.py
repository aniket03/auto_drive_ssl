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

        f_block = torch.cat((fl_patch, f_patch, fr_patch), 3)
        b_block = torch.cat((br_patch, b_patch, bl_patch), 3)
        b_block = torch.flip(b_block, [2,3])

        c_block = torch.cat((f_block, b_block), 2)

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


def get_base_resnet_module(model_type, requires_avg_pool=True):
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
    if requires_avg_pool:
        base_resnet_module = nn.Sequential(*list(original_model.children())[:-1])
    else:
        base_resnet_module = nn.Sequential(*list(original_model.children())[:-2])

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


class FCNResnet(nn.Module):
    def __init__(self, resnet_module, n_classes):
        super(FCNResnet, self).__init__()
        self.resnet_module = resnet_module

        self.n_class = n_classes
        self.pretrained_net = resnet_module
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, n_classes, kernel_size=1)


    def forward(self, x):
        """
        :param x: Batch of images (or pseudo-images - output of aux model)
        """

        output = self.resnet_module(x)                    # size=(N, 512, x.H/32, x.W/32)

        score = self.bn1(self.relu(self.deconv1(output))) # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return nn.functional.sigmoid(score)


def fcn_resnet(model_type, n_classes):
    """
    Returns a FCN-32s network with backbone belonging to the family of ResNets
    :param model_type: Specifies which resnet network to employ. Can be one of {res18, res34, res50}
    :param n_classes: no of classes in the data
    """

    base_resnet_module = get_base_resnet_module(model_type, requires_avg_pool=False)

    return FCNResnet(base_resnet_module, n_classes)



if __name__ == '__main__':

    # Test Combine and up sample
    cm = CombineAndUpSample(64)
    patches_batch = torch.randn(32, 6, 3, 64, 64)
    result = cm.forward(patches_batch)
    print (result.shape)
    del cm, patches_batch, result

    # Test SimCLR model
    sr = simclr_resnet('res18', non_linear_head=False)  # non_linear_head can be True or False either.
    image_batch = torch.randn(32, 3, 64, 64)
    tr_img_batch = torch.randn(32, 3, 64, 64)

    result1, result2 = sr.forward(image_batch, tr_img_batch)

    print (result1.size())
    print (result2.size())
    del sr, image_batch, tr_img_batch, result1, result2

    # Test fcn_resnet with res34
    fr = fcn_resnet('res34', 2)
    image_batch = torch.randn(4, 3, 800, 800)
    result = fr.forward(image_batch)
    print (result.size())
