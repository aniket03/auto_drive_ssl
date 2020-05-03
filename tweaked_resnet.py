from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls


class TweakedResNet(ResNet):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(TweakedResNet, self).__init__(
            block, layers, num_classes, zero_init_residual,
            groups, width_per_group, replace_stride_with_dilation,
            norm_layer)
        del self.fc, self.avgpool

    def forward(self, x):

        results_dict = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        results_dict['res_layer1'] = x
        x = self.layer2(x)
        results_dict['res_layer2'] = x
        x = self.layer3(x)
        results_dict['res_layer3'] = x
        x = self.layer4(x)
        results_dict['res_layer4'] = x

        return results_dict


def _tweaked_resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = TweakedResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model



def tweaked_resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _tweaked_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                           **kwargs)


def tweaked_resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _tweaked_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                           **kwargs)


def tweaked_resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _tweaked_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                           **kwargs)
