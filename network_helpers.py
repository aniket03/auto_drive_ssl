import torch

from models import classifier_resnet, pirl_resnet


def test_copy_weights_resnet_module(m1, m2):
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
        if m1_layer_names[ind][:6] == 'resnet':
            if not torch.all(torch.eq(m1_state_dict[m1_layer_names[ind]].data, m2_state_dict[m2_layer_names[ind]].data)):
                weight_copy_flag = 0
                print ('Something is incorrect for layer {} and {}'.format(m1_layer_names[ind], m2_layer_names[ind]))

    if weight_copy_flag == 1:
        print ('All is well')


def copy_weights_between_models(m1, m2):
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
        if m1_layer_names[ind][:6] == 'resnet':
            cnt += 1
            m2_state_dict[m2_layer_names[ind]] = m1_state_dict[m1_layer_names[ind]].data

    m2.load_state_dict(m2_state_dict)

    print ('Count of layers whose weights were copied between two models', cnt)
    return m2

if __name__ == '__main__':

    pr = pirl_resnet('res18', non_linear_head=False)
    cr = classifier_resnet('res18', num_classes=10)

    copy_success = copy_weights_between_models(pr, cr)

