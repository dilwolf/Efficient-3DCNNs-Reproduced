'''ShuffleNetV2 in PyTorch.
See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
'''

import torch
import torch.nn as nn
from torch.nn import init


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    '''Channel shuffle: [B,C,D,H,W] -> [B,g,C/g,D,H,W] -> [B,C/g,g,D,H,W] -> [B,C,D,H,W]'''
    x = x.view(x.shape[0], groups, x.shape[1] // groups, x.shape[2], x.shape[3], x.shape[4])
    x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
    x = x.view(x.shape[0], -1, x.shape[3], x.shape[4], x.shape[5])
    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.stride == 1:
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv3d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm3d(inp),
                # pw-linear
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :(x.shape[1] // 2), :, :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif self.stride == 2:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=600, sample_size=224, width_mult=1.0):
        super(ShuffleNetV2, self).__init__()
        assert sample_size % 16 == 0

        self.stage_repeats = [4, 8, 4]

        if width_mult == 0.25:
            self.stage_out_channels = [-1, 24,  32,  64, 128, 1024]
        elif width_mult == 0.5:
            self.stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise ValueError("Unsupported width_mult. Choose from: 0.25, 0.5, 1.0, 1.5, 2.0")

        # First layer
        input_channel = self.stage_out_channels[1]
        self.conv1    = conv_bn(3, input_channel, stride=(1, 2, 2))
        self.maxpool  = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Inverted residual blocks
        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat      = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)

        # Last conv
        self.conv_last  = conv_1x1x1_bn(input_channel, self.stage_out_channels[-1])

        # Global average pooling — handles any input size cleanly
        self.avgpool    = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Classifier head
        self.classifier = nn.Linear(self.stage_out_channels[-1], num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.features(out)
        out = self.conv_last(out)
        out = self.avgpool(out)
        out = out.flatten(1)          # (B, C)
        out = self.classifier(out)    # (B, num_classes)
        return out


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = ['classifier']
        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: expected 'complete' or 'last_layer'")


def get_model(**kwargs):
    model = ShuffleNetV2(**kwargs)
    return model