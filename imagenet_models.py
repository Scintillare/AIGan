import torch.nn as nn
import torch.nn.functional as F

# Discriminator - PatchGAN from pix2pix
class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        model = [
            #c8
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # c16
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            # c32
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
        ]
        self.model = nn.Sequential(*model)
        # self.fc = nn.Linear(32*35*35, 5)
        self.fc = nn.Linear(32*35*35, 1)
        self.prob = nn.Sigmoid()

    def forward(self, x):
        output = self.model(x).squeeze()
        output = output.view(output.size(0), -1)
        logits = self.fc(output)
        probs = self.prob(logits)
        return logits, probs



# def ConvInstRelu(f1, f2, ks, s):
#     conv_block = [nn.Conv2d(f1, f2, kernel_size=ks, stride=s, padding=0, bias=True),
#             nn.InstanceNorm2d(f2),
#             nn.ReLU(),]

#     return nn.Sequential(*conv_block)

# def DeConvInstRelu(f1, f2, ks, s):
#     deconv_block = [nn.ConvTranspose2d(f1, f2, kernel_size=ks, stride=s, padding=0, bias=False),
#             nn.InstanceNorm2d(f2),
#             nn.ReLU(),]

#     return nn.Sequential(*deconv_block)

class Generator(nn.Module):
    '''
	Generator definition for AdvGAN
	ref: https://arxiv.org/pdf/1801.02610.pdf
    '''
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()

        encoder_lis = [
            # c7s1-8 3*299*299
            nn.Conv2d(gen_input_nc, 8, kernel_size=7, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # d16 8*293*293
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # d32 16*145*145
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # d64 32*71*71
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # d64 64*34*34
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # d64 64*30*30
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # d64 64*26*26
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 64*22*22
        ]

        #r64 * 4 
        bottle_neck_lis = [ResnetBlock(64),
                       ResnetBlock(64),
                       ResnetBlock(64),
                       ResnetBlock(64),]
        
        decoder_lis = [
            # u64 64*22*22
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # u64 64*26*26
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # u64 64*30*30
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # u32 64*34*34
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # u16 32*71*71
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # u8 16*145*145
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # c7s1-3 8*293*293
            nn.ConvTranspose2d(8, image_nc, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Tanh()
            # 3*299*299
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

        # alternative
        # self.c1 = ConvInstRelu(3, 8, 7, 1)
        # self.d16 = ConvInstRelu(8, 16, 5, 2)
        # self.d32 = ConvInstRelu(16, 32, 5, 2)
        # self.d64 = ConvInstRelu(32, 64, 5, 2)
        # self.d264 = ConvInstRelu(64, 64, 5, 1)

        # self.u64 = DeConvInstRelu(64, 64, 5, 1)
        # self.u32 = DeConvInstRelu(64, 32, 5, 2)
        # self.u16 = DeConvInstRelu(32, 16, 5, 2)
        # self.u8 = DeConvInstRelu(16, 8, 5, 2)
        # self.u3 = nn.ConvTranspose2d(8, 3, kernel_size=7, stride=1, padding=0, bias=False)
        # self.t = nn.Tanh()

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x

# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
