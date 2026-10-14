import torch
import timm
from torch import nn
from torch.nn import functional as F

import torchvision.models as models
from model import mnasnet
from model.utils import *
import model.utils as mu


class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()
        self.in_planes = 0

    def forward(self, x):
        raise NotImplementedError

    def _get_mobilenetv2(self):
        model = models.mobilenet_v2(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        self.in_planes = 1280
        return model

    def _get_mobilenetv2_075(self):
        model = models.mobilenet_v2(pretrained=False, width_mult=0.75)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        self.in_planes = 1280
        return model

    def _get_mobilenetv2_05(self):
        model = models.mobilenet_v2(pretrained=False, width_mult=0.5)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        self.in_planes = 1280
        return model

    def _get_mnasnet0_5(self):
        model = mnasnet.mnasnet0_5(pretrained=True)
        #model = torch.nn.Sequential(*list(model.children())[:-1])
        model = torch.nn.Sequential(*list(next(model.children()))[:-3])  # Remove classification layers.
        self.in_planes = 160
        return model

    def _get_mnasnet1_0(self):
        model = mnasnet.mnasnet1_0(pretrained=True)
        #model = torch.nn.Sequential(*list(model.children())[:-1])
        model = torch.nn.Sequential(*list(next(model.children()))[:-3]) # Remove classification layers.
        self.in_planes = 320
        return model

    def _get_mobilenetv3_s_0_75(self):
        model = self._load_timm_model('tf_mobilenetv3_small_075')
        model = torch.nn.Sequential(*list(model.children())[:-4])
        self.in_planes = 432
        return model

    def _get_mobilenetv3_l_0_75(self):
        model = self._load_timm_model('tf_mobilenetv3_large_075')
        model = torch.nn.Sequential(*list(model.children())[:-4])
        self.in_planes = 720
        return model

    def _get_mobilenetv3_s_1_0(self):
        model = self._load_timm_model('tf_mobilenetv3_small_100')
        model = torch.nn.Sequential(*list(model.children())[:-4])
        self.in_planes = 576
        return model

    def _get_mobilenetv3_l_1_0(self):
        model = self._load_timm_model('tf_mobilenetv3_large_100')
        model = torch.nn.Sequential(*list(model.children())[:-4])
        self.in_planes = 960
        return model

    def _get_efficientnet_b0(self):
        model = self._load_timm_model('efficientnet_b0')
        model = torch.nn.Sequential(*list(model.children())[:-4])
        # model = model.features
        self.in_planes = 1280
        return model

    def _load_timm_model(self, name):
        try:
            import timm
            model = timm.create_model(name, pretrained=not self.exportable, scriptable=True, exportable=self.exportable)
        except:
            model = torch.hub.load('rwightman/pytorch-image-models', name, pretrained=True)
        return model

    def _get_mixnet_s(self):
        model = self._load_timm_model('mixnet_s')
        model = torch.nn.Sequential(*list(model.children())[:-5])  # Remove classification layers.
        self.in_planes = 200
        return model

    def _get_mixnet_s_05(self):
        import timm.models.efficientnet as e
        import timm.models.layers.config as c
        c.set_exportable(self.exportable)
        c.set_scriptable(True)
        model = e._gen_mixnet_s('mixnet_s', channel_multiplier=0.5, pretrained=False)
        model = torch.nn.Sequential(*list(model.children())[:-5])  # Remove classification layers.
        self.in_planes = 104
        return model

    def _get_mixnet_m(self):
        model = self._load_timm_model('mixnet_m')
        model = torch.nn.Sequential(*list(model.children())[:-2])
        self.in_planes = 1536
        return model

    def _get_mixnet_l(self):
        model = self._load_timm_model('mixnet_l')
        model = torch.nn.Sequential(*list(model.children())[:-2])
        self.in_planes = 1536
        return model

    def _get_features_model(self, name, exportable=False):
        """
        Return a muli scale features model ( instead of only last layers, can be used to get information for multi-scale networks ).
        Currently support many backbone.
        See https://rwightman.github.io/pytorch-image-models/feature_extraction/
        """
        import timm
        model = timm.create_model(name, pretrained=not exportable, features_only=True, exportable=exportable,
                                  no_jit=exportable)
        return model

    def _get_resnet_18(self):
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-2])
        self.in_planes = 512
        return model

    def _get_resnet_50(self):
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-2])
        self.in_planes = 2048
        return model

    def _get_resnet_101(self):
        model = models.resnet101(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-2])
        self.in_planes = 2048
        return model

    def _get_resnet_152(self):
        model = models.resnet152(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-2])
        self.in_planes = 2048
        return model

    def _get_backbone(self, backbone_name, pretrained, all_features=False):

        backbone = None
        if backbone_name == "mnasnet0_5":
            backbone = self._get_mnasnet0_5()
        elif backbone_name == "mnasnet1_0":
            backbone = self._get_mnasnet1_0()
        elif backbone_name == "mobilenetv2":
            backbone = self._get_mobilenetv2()
        elif backbone_name == "mobilenetv2_075":
            backbone = self._get_mobilenetv2_075()
        elif backbone_name == "mobilenetv2_05":
            backbone = self._get_mobilenetv2_05()
        elif backbone_name == 'mobilenetv3_s_0_75':
            backbone = self._get_mobilenetv3_s_0_75()
        elif backbone_name == 'mobilenetv3_l_0_75':
            backbone = self._get_mobilenetv3_l_0_75()
        elif backbone_name == 'mobilenetv3_s_1_0':
            backbone = self._get_mobilenetv3_s_1_0()
        elif backbone_name == 'mobilenetv3_l_1_0':
            backbone = self._get_mobilenetv3_l_1_0()
        elif backbone_name == 'efficientnet_b0':
            backbone = self._get_efficientnet_b0()
        elif backbone_name == 'ghostnet':
            backbone = self._get_ghostnet()
        elif backbone_name == 'mixnet_s':
            backbone = self._get_mixnet_s()
        elif backbone_name == 'mixnet_s_05':
            backbone = self._get_mixnet_s_05()
        elif backbone_name == 'mixnet_m':
            backbone = self._get_mixnet_m()
        elif backbone_name == 'mixnet_l':
            backbone = self._get_mixnet_l()
        elif backbone_name == 'resnet_18':
            backbone = self._get_resnet_18()
        elif backbone_name == 'resnet_50':
            backbone = self._get_resnet_50()
        elif backbone_name == 'resnet_101':
            backbone = self._get_resnet_101()
        elif backbone_name == 'resnet_152':
            backbone = self._get_resnet_152()
        elif backbone_name.startswith('timm_'):
            name = backbone_name[5:]
            backbone = timm.create_model(name, pretrained=pretrained, features_only=True, exportable=True, no_jit=True)
            channels = backbone.feature_info[-1]['num_chs']
            self.feature_info = backbone.feature_info
            self.in_planes = channels
        else:
            raise Exception('Unknown backbone: ', backbone_name)

        return backbone


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


class CustomLinearBottleneck(nn.Module):
    def __init__(self, inp, mid, oup, stride, norm_layer=None):
        super(CustomLinearBottleneck, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = mid
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []

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


class UpsamplingLayer(nn.Module):
    def __init__(self, n_in, n_out, method="biconv", scale=2.0, group=1, act_method="relu6", last_act=True):
        super(UpsamplingLayer, self).__init__()
        if act_method == "relu6":
            act_layer = nn.ReLU6(inplace=True)
        elif act_method == "relu":
            act_layer = nn.ReLU(inplace=True)
        else:
            act_layer = nn.ReLU(inplace=True)

        layers = []
        if method == "biconv":
            layers.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(n_in, n_out, 3, 1, 1, groups=1, bias=False, dilation=1))
        elif method == "convtrans":
            layers.append(nn.ConvTranspose2d(n_in, n_out, 4, 2, padding=1, groups=group, bias=False))
        elif method == "convtransfull":
            layers.append(nn.ConvTranspose2d(n_in, n_out, 4, 2, padding=1, groups=1, bias=False))
        elif method == "upsample":
            layers.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(n_in, n_out, 3, 1, 1, groups=group, bias=False))
        elif method == "upsample_align":
            layers.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True))
            layers.append(nn.Conv2d(n_in, n_out, 3, 1, 1, groups=group, bias=False))
        elif method == "upsample_align_dwsep":
            layers.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True))
            layers.append(DWConvLayer(n_in, n_out, 1, act_method=act_method, last_act=False))
        elif method == "upsample_dwsep":
            layers.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))
            layers.append(DWConvLayer(n_in, n_out, 1, act_method=act_method, last_act=False))
        elif method == "upsample_dw":
            layers.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(n_in, n_in, 3, 1, 1, groups=n_in, bias=False))
            layers.append(nn.BatchNorm2d(n_in))
            layers.append(act_layer)
            # pointwise
            layers.append(nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, padding=0))
        elif method == "upsample_1":
            layers.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(n_in, n_out, 1, 1, 0, groups=1, bias=False))
        elif method == "upsample_light":
            layers.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(n_in, n_out, 1, 1, 0, groups=group, bias=False))
        elif method == "pixelshuffle":
            layers.append(nn.PixelShuffle(int(scale)))
            n_out = int(n_out / (scale * scale))
        else:
            print("Unimplemented function method is selected!")
            exit(1)

        if last_act:
            layers.append(nn.BatchNorm2d(n_out))
            layers.append(act_layer)

        self.conv = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
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