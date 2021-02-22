import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from torch.nn import init
import functools
from activations import *

'''
Models from pix2pix
'''

def get_resnet50(num_classes):
    model = models.resnet50(pretrained=False)
    #instead normalization prepend batchnorm
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, num_classes))#, nn.LogSoftmax(dim=1))
    # model = nn.Sequential(nn.BatchNorm2d(num_features=3, affine=False), model)
    return model

# Discriminator - PatchGAN from pix2pix
class PatchDiscriminator(nn.Module):
    def __init__(self, image_nc):
        super(PatchDiscriminator, self).__init__()
        model = [
            #c8
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=(1, 1), bias=True),
            nn.LeakyReLU(0.2, inplace=True),           
            # Rational(),
            # AReLU(),
            # c16
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=(1, 1), bias=True),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),           
            # Rational(),
            # AReLU(),
            # c32
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=(1, 1), bias=True),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),           
            # Rational(),
            # AReLU(),
            # # c64
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1, 1), bias=True),
            # nn.InstanceNorm2d(32),
            # nn.LeakyReLU(0.2, inplace=True),           
            # Rational(),
            # AReLU(),

            # nn.Conv2d(64, 1, 1, bias=True),
            nn.Conv2d(32, 1, 1, bias=True),
        ]
        self.model = nn.Sequential(*model)
        self.prob = nn.Sigmoid()

    def forward(self, x):
        output = self.model(x).squeeze() 
        probs = self.prob(output)
        return output, probs

# pix2pix
class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.InstanceNorm2d): #norm_layer=nn.BatchNorm2d
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)
        self.prob = nn.Sigmoid()

    def forward(self, input):
        """Standard forward."""
        output = self.net(input)
        return output, self.prob(output)

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)




class ResnetGenerator(nn.Module):
    '''
    For 299*299 images.
    '''
    
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(ResnetGenerator, self).__init__()

        encoder_lis = [
            # c7s1-8 3*299*299
            nn.Conv2d(gen_input_nc, 8, kernel_size=7, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            # d16 8*293*293
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            # d32 16*145*145
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            # d64 32*71*71
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            # d64 64*34*34
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            # d64 64*30*30
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            # d64 64*26*26
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
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
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            # u64 64*26*26
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(64),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            # u64 64*30*30
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(64),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            # u32 64*34*34
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(32),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            # u16 32*71*71
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            # u8 16*145*145
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            # c7s1-3 8*293*293
            nn.ConvTranspose2d(8, image_nc, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Tanh()
            # 3*299*299
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

        

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x

def ConvInstRelu(f1, f2, ks, s, p=0):
    conv_block = [nn.Conv2d(f1, f2, kernel_size=ks, stride=s, padding=p, bias=True),
            nn.InstanceNorm2d(f2),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),            
            # Rational(),
            # AReLU(),
            ]

    return nn.Sequential(*conv_block)

def DeConvInstRelu(f1, f2, ks, s, p=0):
    deconv_block = [nn.ConvTranspose2d(f1, f2, kernel_size=ks, stride=s, padding=p, bias=False),
            nn.InstanceNorm2d(f2),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),           
            # Rational(),
            # AReLU(),
            ]

    return nn.Sequential(*deconv_block)


class Resnet224Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Resnet224Generator, self).__init__()

        self.c1 = ConvInstRelu(3, 8, 7, 1, 1)
        self.d16 = ConvInstRelu(8, 16, 4, 2, 1)
        self.d32 = ConvInstRelu(16, 32, 4, 2, 1)
        self.d64 = ConvInstRelu(32, 64, 3, 2, 1)
        self.d264 = ConvInstRelu(64, 64, 3, 1)
        # self.d128 = ConvInstRelu(64, 128, 4, 2, 1)
        # self.d2128 = ConvInstRelu(128, 128, 3, 1)

        #r64 * 4 
        bottle_neck_lis = [ResnetBlock(64),
                       ResnetBlock(64),
                       ResnetBlock(64),
                       ResnetBlock(64),]
        # bottle_neck_lis = [ResnetBlock(128),
        #                ResnetBlock(128),
        #                ResnetBlock(128),
        #                ResnetBlock(128),]
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)

        self.u64 = DeConvInstRelu(64, 64, 3, 1)
        # self.u128 = DeConvInstRelu(128, 128, 3, 1)
        # self.u64 = DeConvInstRelu(128, 64, 4, 2, 1)
        self.u32 = DeConvInstRelu(64, 32, 3, 2, 1)
        self.u16 = DeConvInstRelu(32, 16, 4, 2, 1)
        self.u8 = DeConvInstRelu(16, 8, 4, 2, 1)
        self.u3 = nn.ConvTranspose2d(8, 3, kernel_size=7, stride=1, padding=1, bias=False)
        self.t = nn.Tanh()

    def forward(self, x):
        x = self.c1(x) # 8, 220, 220
        x = self.d16(x) # 16, 110, 110
        x = self.d32(x) # 32, 55, 55
        x = self.d64(x) # 64, 28, 28
        x = self.d264(x) # 64, 26, 26
        x = self.d264(x) # 64, 24, 24
        x = self.d264(x) # 64, 22, 22
        # x = self.d128(x)
        # x = self.d2128(x)
        # x = self.d2128(x)
        # x = self.d2128(x)
        x = self.bottle_neck(x) # 64, 22, 22
        x = self.u64(x) # 64, 24, 24
        x = self.u64(x)# 64, 26, 26
        x = self.u64(x) # 64, 28, 28
        # x = self.u128(x)
        # x = self.u128(x)
        # x = self.u128(x)        
        # x = self.u64(x)
        x = self.u32(x) # 32, 55, 55 
        x = self.u16(x) # 16, 110, 110
        x = self.u8(x) # 8, 220, 220
        x = self.u3(x) # 3, 224, 224
        x = self.t(x) # 3, 224, 224
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
                       nn.ReLU(True)
                    # Rational(),
                    # AReLU(),
                       ]
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


# Define a unet block
# from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)