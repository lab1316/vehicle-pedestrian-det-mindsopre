from mindcv.models import create_model
import mindspore as ms
from mindspore import nn, ops
import math


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


'''
taken from https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py


modifications: remove final classification layers, fix missing keys in state dicts for transfer learning
-> ready to plug in as backbone for a detection head

use width_mult to change width of the model, can't do transfer learning this way though
'''


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.SequentialCell):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=-1, bias=False):
        if padding == -1:
            padding = (kernel_size - 1) // 2
        # super(ConvBNReLU, self).__init__()
        cov = ms.nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, pad_mode="pad",group=groups, has_bias=False)
        norm = ms.nn.BatchNorm2d(out_planes)
        relu = ms.nn.ReLU6()
        super(ConvBNReLU, self).__init__(nn.SequentialCell(cov,norm,relu))

class InvertedResidual(nn.Cell):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = nn.SequentialCell()
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
       
            # dw
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim)),
            # pw-linear
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, padding=0, has_bias=False))
        layers.append(nn.BatchNorm2d(oup))
        self.conv = layers
       

    def construct(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Cell):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()
        self.num_classes = num_classes

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        if self.num_classes != 2:
            # building last several layers
            features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))

        # make it nn.Sequential
        self.features = nn.SequentialCell(*features)

        # weight initialization
        for _,m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(ms.common.initializer.initializer(
            ms.common.initializer.HeNormal(negative_slope=0, mode='fan_out'),
            m.weight.shape, m.weight.dtype))
                if m.bias is not None:
                    m.bias.set_data(ms.common.initializer.initializer("zeros", m.bias.shape, m.bias.dtype))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(ms.common.initializer.initializer("ones", m.gamma.shape, m.gamma.dtype))
                m.beta.set_data(ms.common.initializer.initializer("zeros", m.beta.shape, m.beta.dtype))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(ms.common.initializer.initializer(
            ms.common.initializer.HeUniform(negative_slope=math.sqrt(5)),
            m.weight.shape, m.weight.dtype))
                m.bias.set_data(ms.common.initializer.initializer("zeros", m.bias.shape, m.bias.dtype))


    def _construct(self, x):
        for idx, layer in enumerate(self.features):
            # want to get expansion of layer 15
            if idx == 14:
                res_connect = x
                for i, element in enumerate(layer.conv):
                    x = element(x)
                    if self.num_classes == 2:
                        inter = res_connect
                    else:
                        if i == 0:
                            inter = x
                if layer.use_res_connect:
                    x += res_connect
            else:
                x = layer(x)
        return inter, x

    # Allow for accessing forward method in a inherited class
    construct = _construct


def mobilenet_v2(pretrained=True, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        pre_save_path = "11111.ckpt"
        network = create_model(model_name='mobilenet_v2_140', num_classes=81, pretrained=True)
        ms.save_checkpoint(network, pre_save_path)
        pretrained_dict = ms.load_checkpoint(pre_save_path)
        # pretrained_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
        #                                            progress=progress)
        msp = "2222.ckpt"
        ms.save_checkpoint(model,msp)
        model_dict = ms.load_checkpoint(msp)
        # only keep keys of the model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        ms.load_param_into_net(model,pretrained_dict)
    return model