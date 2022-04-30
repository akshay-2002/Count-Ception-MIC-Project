import torch
import torch.nn as nn
import torch.nn.init as init

# Convolution block
class ConvolutionBlock(nn.Module):

    # ksize = kernel size
    def __init__(self, input_channels, output_channels, ksize=3, stride=1, pad=0, activation=nn.LeakyReLU()):
        super(ConvolutionBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=ksize, stride=stride, padding=pad)
        self.activation = activation
        self.batch_norm = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv1(x)))

# For the two green blocks in the neural arch
class SimpleBlock(nn.Module):
    def __init__(self, input_channels, output_channels_1x1, output_channels_3x3, activation=nn.LeakyReLU()):
        super(SimpleBlock, self).__init__()
        # We need two blocks 
        self.conv1 = ConvolutionBlock(input_channels, output_channels_1x1, ksize=1, pad=0, activation=activation)
        self.conv2 = ConvolutionBlock(input_channels, output_channels_3x3, ksize=3, pad=1, activation=activation)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        concatenated_block = torch.cat([conv1_out, conv2_out], 1)
        return concatenated_block

class ModelCountception(nn.Module):
    def __init__(self, inplanes=3, outplanes=1, use_logits=False, logits_per_output=12, debug=False):
        super(ModelCountception, self).__init__()
        # params
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.activation = nn.LeakyReLU(0.01)
        self.final_activation = nn.LeakyReLU(0.01)
        # receptive field size
        self.patch_size = 32
        self.use_logits = use_logits
        self.logits_per_output = logits_per_output
        self.debug = debug

        torch.LongTensor()

        # Defining the green block
        self.conv1 = ConvolutionBlock(self.inplanes, 64, ksize=3, pad=self.patch_size, activation=self.activation)
        self.simple1 = SimpleBlock(64, 16, 16, activation=self.activation)
        self.simple2 = SimpleBlock(32, 16, 32, activation=self.activation)
        self.conv2 = ConvolutionBlock(48, 16, ksize=14, activation=self.activation)
        self.simple3 = SimpleBlock(16, 112, 48, activation=self.activation)
        self.simple4 = SimpleBlock(160, 64, 32, activation=self.activation)
        self.simple5 = SimpleBlock(96, 40, 40, activation=self.activation)
        self.simple6 = SimpleBlock(80, 32, 96, activation=self.activation)
        self.conv3 = ConvolutionBlock(128, 32, ksize=18, activation=self.activation)
        self.conv4 = ConvolutionBlock(32, 64, ksize=1, activation=self.activation) 
        self.conv5 = ConvolutionBlock(64, 64, ksize=1, activation=self.activation)
        if use_logits:
            self.conv6 = nn.ModuleList([ConvolutionBlock(
                64, logits_per_output, ksize=1, activation=self.final_activation) for _ in range(outplanes)])
        else:
            self.conv6 = ConvolutionBlock(64, self.outplanes, ksize=1, activation=self.final_activation)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight, gain=init.calculate_gain('leaky_relu', param=0.01))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _print(self, x):
        if self.debug:
            print(x.size())

    def forward(self, x):
        net = self.conv1(x)  # 32
        net = self.simple1(net)
        net = self.simple2(net)
        net = self.conv2(net)
        net = self.simple3(net)
        net = self.simple4(net)
        net = self.simple5(net)
        net = self.simple6(net)
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.conv5(net)
        if self.use_logits:
            net = [c(net) for c in self.conv6]
        else:
            net = self.conv6(net)
        return net

    def name(self):
        return 'countception'
