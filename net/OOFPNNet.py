from torch import nn
from model.layers import *
from model.backbones import Backbone
import math

class Network(Backbone):
    def __init__(self, conf=None):
        super(Network, self).__init__()

        self.pretrained = True
        self.align_corners = False

        encoder = self._get_backbone("resnet_18", self.pretrained, all_features=True)
        encoder_list = list(encoder.children())
        self.base1 = torch.nn.Sequential(*encoder_list[:-3]) # resnet18 layer1
        self.base2 = torch.nn.Sequential(*encoder_list[-3:-2]) # resnet18 layer2

        # 16 32 64 128
        decoder_ch = 256

        self.proj_f1 = nn.Conv2d(64, decoder_ch, 1)

        self.decoder1 = nn.Sequential(DWSConv(128, decoder_ch, 3),
                                      DWSConv(decoder_ch, decoder_ch, 3),
                                      DWSConv(decoder_ch, decoder_ch, 3))

        self.decoder2 = nn.Sequential(DWSConv(decoder_ch, decoder_ch, 3),
                                      DWSConv(decoder_ch, decoder_ch, 3),
                                      DWSConv(decoder_ch, decoder_ch, 3))

        self.predict = DepthWiseSepConv(decoder_ch, 5, 3, act_method=None, norm_layer=None, bias=True)

        self.up = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=self.align_corners)

    def forward(self, x):
        feature1 = self.base1(x)  # 128x128 -> 32x32
        feature2 = self.base2(feature1)  # 32x32 -> 16x16

        feature_decode1 = self.decoder1(feature2)  # 16x16 -> 32x32
        feature_decode1_up = self.up(feature_decode1) + self.proj_f1(feature1)
        feature_decode2 = self.decoder2(feature_decode1_up)  # 32x32 -> 64x64

        y = self.predict(feature_decode2)

        return y

if __name__ == '__main__':
    from utils.model_utils import print_flops, print_model_summary
    from rf_calc import receptive_field
    import logging
    import time

    logging.basicConfig(level=logging.INFO)

    device = 'cpu'
    model = Network(conf=None).to(device)

    input_shape = (1, 3, 256, 384) # output 12
    # 360 640
    # 180 320

    input = torch.zeros(size=input_shape).to(device)

    start_time = time.perf_counter()
    output = model(input)
    end_time = time.perf_counter()

    print((end_time - start_time) * 1000, 'ms')

    def print_output(output):
        if isinstance(output, torch.Tensor):
            print('output.shape')
            print(output.shape)
        elif isinstance(output, list) or isinstance(output, tuple):
            for i, tensor in enumerate(output):
                print('output', i, ': ', tensor.shape)

    print_output(output)

    #RF = receptive_field(model, 640)

    #print("Receptive Field: ", RF)

    print_flops(model, input_shape=input_shape)
    #print_model_summary(model, input_shape=(1, 3, 256, 256))

    model_name = __file__[:-3].replace("net", "onnx__")
    torch.onnx.export(model,
                      input,
                      f'{model_name}.onnx',
                      export_params=True,
                      verbose=False,
                      input_names=['image'],
                      #output_names=['refine_out', 'global_out1', 'global_out2', 'global_out3'], opset_version=11)
                      output_names=['refine_out'], opset_version=11)