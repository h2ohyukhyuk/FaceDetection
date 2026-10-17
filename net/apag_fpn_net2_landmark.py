"""APAG FPN2: 5 detection + 5 landmark heatmap + 10 local offset channels.

Landmark order: image-left eye, image-right eye, nose, image-left mouth,
image-right mouth. Offsets are in stride-4 grid units at landmark locations.
"""
import torch
from torch import nn
from net.apag_fpn_net2 import Network as Detector,fpn_ch

class Network(Detector):
    def __init__(self,conf=None):
        super().__init__(conf)
        self.landmark_head=nn.Sequential(
            nn.Upsample(scale_factor=2.,mode='bilinear',align_corners=False),
            nn.Conv2d(fpn_ch,fpn_ch,3,padding=1,groups=fpn_ch,bias=False),
            nn.Conv2d(fpn_ch,15,1))
        nn.init.zeros_(self.landmark_head[-1].bias)

    def forward(self,x):
        x1=self.layer1(x);x2=self.layer2(x1)
        c1=self.layer2_extra(x2)
        x3=self.layer3(x2);c2=self.layer3_extra(x3)
        x4=self.layer4(x3);c3=self.layer4_extra(x4)
        c4=self.layer5(x4)
        p1=self.fpn(c1,c2,c3,c4)
        return torch.cat((self.head1(p1),self.landmark_head(p1)),dim=1)
