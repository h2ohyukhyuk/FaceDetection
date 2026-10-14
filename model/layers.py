
import torch
from torch import nn
from torch.nn import functional as F

class ConvReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, act_method="relu6", bias=False):
        padding = (kernel_size - 1) // 2

        if act_method == "relu6":
            act_layer = nn.ReLU6(inplace=True)
        elif act_method == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif act_method == "prelu":
            act_layer = nn.PReLU()
        else:
            act_layer = nn.ReLU(inplace=True)

        super(ConvReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias),
            act_layer
        )

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, act_method="relu6", norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if act_method == "relu6":
            act_layer = nn.ReLU6(inplace=True)
        elif act_method == "relu":
            act_layer = nn.ReLU(inplace=True)
        else:
            act_layer = nn.ReLU(inplace=True)

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            act_layer
        )


class LinearBottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(LinearBottleneck, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DWConvLayer(nn.Module):
    def __init__(self, n_in, n_out, stride=1, pooling=None, last_act=True, act_method="relu6"):
        super(DWConvLayer, self).__init__()
        layers = []
        # depthwise
        layers.append(
            nn.Conv2d(n_in,
                      n_in,
                      kernel_size=3,
                      padding=1,
                      stride=stride,
                      groups=n_in))
        layers.append(nn.BatchNorm2d(n_in))
        if act_method == "relu6":
            layers.append(nn.ReLU6(inplace=True))
        elif act_method == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif act_method == "prelu":
            layers.append(nn.PReLU())
        else:
            layers.append(nn.ReLU(inplace=True))

        # pointwise
        layers.append(nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, padding=0))

        if pooling is not None:
            if pooling == "max":
                layers.append(nn.MaxPool2d(kernel_size=2))
            elif pooling == "avg":
                layers.append(torch.nn.AvgPool2d(kernel_size=2))

        if last_act:
            layers.append(nn.BatchNorm2d(n_out))
            if act_method == "relu6":
                layers.append(nn.ReLU6(inplace=True))
            elif act_method == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif act_method == "prelu":
                layers.append(nn.PReLU())
            else:
                layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.conv(x)

class ConvBNNL(nn.Sequential):
    # conv -> bn -> non-linearity
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, act_method='relu'):
        # default non-linearity : ReLU6
        padding = (kernel_size - 1) // 2

        if act_method == "relu6":
            super(ConvBNNL, self).__init__(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU6(inplace=True))
        elif act_method == "prelu":
            super(ConvBNNL, self).__init__(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes))
        else:
            super(ConvBNNL, self).__init__(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True))

class DepthWiseSepConv(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, act_method='relu', norm_layer=nn.BatchNorm2d, bias=False, residual=True):
        super(DepthWiseSepConv, self).__init__()

        self.inp = inp
        self.oup = oup
        self.residual = residual

        layers = []
        layers.extend([
            # dw
            ConvBNNL(inp, inp, kernel_size, stride, groups=inp, act_method=act_method),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=bias),

        ])

        if norm_layer is not None:
            layers.append(norm_layer(oup))

        if act_method == "relu6":
            layers.append(nn.ReLU6(inplace=True))
        elif act_method == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif act_method == "prelu":
            layers.append(nn.PReLU())

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv(x)
        if self.residual and self.inp == self.oup:
            y = y + x
        return y


class ResidualConv(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, bias=False):
        super(ResidualConv, self).__init__()

        self.inp = inp
        self.oup = oup

        padding = (kernel_size - 1) // 2

        layers = []
        layers.extend([
            nn.Conv2d(inp, inp, kernel_size, 1, padding, bias=bias),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, kernel_size, 1, padding, bias=bias),
            nn.BatchNorm2d(oup)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv(x)

        if self.inp == self.oup:
            y = y + x

        y = F.relu(y)

        return y

class ResidualConv1(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, bias=False):
        super(ResidualConv1, self).__init__()

        self.inp = inp
        self.oup = oup

        padding = (kernel_size - 1) // 2

        layers = []
        layers.extend([
            nn.Conv2d(inp, oup, kernel_size, 1, padding, bias=bias),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv(x)

        if self.inp == self.oup:
            y = y + x

        y = F.relu(y)

        return y

class DWSConv(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, act_method='relu', norm_layer=nn.BatchNorm2d, bias=False, residual=True):
        super(DWSConv, self).__init__()

        self.inp = inp
        self.oup = oup
        self.residual = residual
        self.act_method = act_method

        layers = []
        layers.extend([
            # dw
            ConvBNNL(inp, inp, kernel_size, stride, groups=inp, act_method=act_method),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=bias),

        ])

        if norm_layer is not None:
            layers.append(norm_layer(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv(x)
        if self.residual and self.inp == self.oup:
            y = y + x

        if self.act_method == "relu6":
            y = F.relu6(y)
        elif self.act_method == "relu":
            y = F.relu(y)
        elif self.act_method == "prelu":
            y = F.prelu(y)

        return y