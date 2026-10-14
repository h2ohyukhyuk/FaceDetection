import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import ResidualConv1

class Network(nn.Module):
    def __init__(self, conf=None):
        super(Network, self).__init__()

        num_classes = 5

        # 256 -> 64
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, stride=2),
                                    nn.Tanh(),
                                    DSConv(16, 64),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))

        # 64 -> 32
        self.layer2 = nn.Sequential(DSConv(64, 64),
                                    nn.ReLU(inplace=True),
                                    DSConv(64, 64),
                                    nn.ReLU(inplace=True),
                                    DSConv(64, 64),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))

        self.layer2_extra = nn.Sequential(ResidualConv1(64,64,1, False),
                                          ResidualConv1(64,64,1, False),
                                          DSConv(64, 128),
                                          nn.ReLU(inplace=True))

        self.layer2_predict = nn.Sequential(nn.Conv2d(128,128,3, padding=1, groups=128,bias=False),
                                            nn.Conv2d(128, num_classes, 1, bias=True))

        # 32 -> 16
        self.layer3 = nn.Sequential(DSConv(64, 128),
                                    nn.ReLU(inplace=True),
                                    DSConv(128, 128),
                                    nn.ReLU(inplace=True),
                                    DSConv(128, 128),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))

        self.layer3_extra = nn.Sequential(ResidualConv1(128,128,1, False),
                                          ResidualConv1(128,128,1, False),
                                          DSConv(128, 128),
                                          nn.ReLU(inplace=True))

        self.layer3_predict = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
                                            nn.Conv2d(128, num_classes, 1, bias=True))

        # 16 -> 8
        self.layer4 = nn.Sequential(DSConv(128, 256),
                                    nn.ReLU(inplace=True),
                                    DSConv(256, 256),
                                    nn.ReLU(inplace=True),
                                    DSConv(256, 256),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))

        self.layer4_extra = nn.Sequential(ResidualConv1(256, 128, 1, False),
                                          ResidualConv1(128, 128, 1, False),
                                          DSConv(128, 128),
                                          nn.ReLU(inplace=True))

        self.layer4_predict = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
                                            nn.Conv2d(128, num_classes, 1, bias=True))

        # 8 -> 4
        self.layer5 = nn.Sequential(DSConv(256, 256),
                                    nn.ReLU(inplace=True),
                                    DSConv(256, 256),
                                    nn.ReLU(inplace=True))

        self.layer5_predict = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=False),
                                            nn.Conv2d(256, num_classes, 1,bias=True))

        self.up = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False)


    def forward(self, x):
        x1 = self.layer1(x) # input: 256x384 <
        x2 = self.layer2(x1)

        y1 = self.layer2_extra(x2)
        y1 = self.layer2_predict(self.up(y1)) # rf: 50x50, out: 64x96, stride: 8, obj: 10~45

        x3 = self.layer3(x2)
        y2 = self.layer3_extra(x3)
        y2 = self.layer3_predict(self.up(y2)) # rf: 135x135, out: 32x48, stride: 16, obj: 40~80

        x4 = self.layer4(x3)
        y3 = self.layer4_extra(x4)
        y3 = self.layer4_predict(self.up(y3)) # rf: 300x300, out: 16x24: stride: 32, obj: 70~140

        x5 = self.layer5(x4)
        y4 = self.layer5_predict(x5) # rf:445 x 445, out: 16x24: stride: 32, obj: 130~384

        return y1, y2, y3, y4

class DSConv(nn.Module):
    def __init__(self, inp, outp, s=1):
        super(DSConv, self).__init__()
        self.inp = inp
        self.outp = outp

        self.conv_dw = nn.Conv2d(inp, inp, (3,3), groups=inp, stride=s, padding=1)
        self.conv_pt1 = nn.Conv2d(inp, inp, 1, stride=s, padding=0)
        self.conv_pt2 = nn.Conv2d(inp, outp, 1, stride=s, padding=0)
        self.bn1 = nn.BatchNorm2d(inp)
        self.bn2 = nn.BatchNorm2d(outp)

        if self.inp != self.outp:
            self.skip = nn.Conv2d(inp, outp,1, groups=inp, stride=s, padding=0)


    def forward(self,x):
        y = self.conv_dw(x)
        y = self.conv_pt1(y)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv_pt2(y)
        y = self.bn2(y)

        if self.inp != self.outp:
            y = y + self.skip(x)
        else:
            y = y + x

        return y



if __name__ == '__main__':

    import yaml
    from utils.model_utils import print_flops, print_model_summary
    import time
    from rf_calc import receptive_field
    from utils.model_utils import load_model_from_path

    param_path = '../experiments/ShiftNet.yml'
    with open(param_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    device = 'cpu'
    model = Network()

    model = load_model_from_path('/home/j/models_cla/apag-imgn/086.pt', model, strict=False)

    model = model.to(device)

    input_shape = (1, 3, 256, 384)  # output 12

    input = torch.zeros(size=input_shape).to(device)

    start_time = time.perf_counter()
    output = model(input)
    end_time = time.perf_counter()

    print((end_time - start_time)*1000, 'ms')


    def print_output(output):
        if isinstance(output, torch.Tensor):
            print('output.shape')
            print(output.shape)
        elif isinstance(output, list) or isinstance(output, tuple):
            for i, tensor in enumerate(output):
                print('output', i, ': ', tensor.shape)

    print_output(output)

    #RF = receptive_field(model, 384)

    print_flops(model, input_shape=input_shape)
    # printmodel_summary(model, input_shape=(1, 3, 256, 256))

    model_name = __file__[:-3].replace("net", "onnx__")
    torch.onnx.export(model,
                      input,
                      f'{model_name}.onnx',
                      export_params=True,
                      verbose=False,
                      input_names=['image'],
                      # output_names=['refine_out', 'global_out1', 'global_out2', 'global_out3'], opset_version=11)
                      output_names=['refine_out'], opset_version=11)

    # import onnx
    # from onnxruntime.quantization import quantize_dynamic, QuantType
    #
    # quantize_dynamic(f'{model_name}.onnx',f'{model_name}_quant.onnx', weight_type=QuantType.QUInt8)
