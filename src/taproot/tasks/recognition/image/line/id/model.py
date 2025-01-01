# Adapted from https://github.com/patrickvonplaten/controlnet_aux/blob/master/src/controlnet_aux/lineart/__init__.py and https://github.com/patrickvonplaten/controlnet_aux/blob/master/src/controlnet_aux/lineart_anime/__init__.py
import functools
import torch
import torch.nn as nn

from typing import Optional, Union, Type, List

class ResidualBlock(nn.Module):
    """
    A residual block that contains two convolutional layers with the same number of input and output channels.
    """
    def __init__(self, in_features: int) -> None:
        """
        :param in_features: The number of input features.
        """
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: The input tensor.
        :return: The output tensor, which is the input tensor plus the result of the convolutional block.
        """
        return x + self.conv_block(x) # type: ignore[no-any-return]

class Generator(nn.Module):
    """
    The generator of the model, which consists of an initial convolution block,
    a downsampling block, a residual block, an upsampling block, and an output layer.
    """
    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        n_residual_blocks: int=9,
        sigmoid: bool=True
    ) -> None:
        super(Generator, self).__init__()
        # Initial convolution block
        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        self.model1 = nn.Sequential(*model1)

        # Residual blocks
        self.model2 = nn.Sequential(*[
            ResidualBlock(in_features) for _ in range(n_residual_blocks)
        ])

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        self.model3 = nn.Sequential(*model3)

        # Output layer
        self.model4 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            (nn.Identity() if not sigmoid else nn.Sigmoid())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: The input tensor, an image of shape (N, C, H, W).
        :return: The output tensor, an image of shape (N, C, H, W).
        """
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)
        return out # type: ignore[no-any-return]

class UNetGenerator(nn.Module):
    """
    Create a UNet-based generator
    """

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        num_downs: int,
        ngf: int=64,
        norm_layer: Union[Type[nn.Module], functools.partial[nn.Module]]=nn.BatchNorm2d,
        use_dropout: bool=False,
    ) -> None:
        """
        Construct a UNet generator

        :param input_nc: the number of channels in input images
        :param output_nc: the number of channels in output images
        :param num_downs: the number of downsamplings in UNet. For example, 
                          if |num_downs image of size 128x128 will become
                          of size 1x1 # at the bottleneck
        :param ngf: the number of filters in the last conv layer
        :param norm_layer: normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UNetGenerator, self).__init__()
        # construct unet structure
        unet_block = UNetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True
        )  # add the innermost layer
        for _ in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UNetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UNetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UNetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UNetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer
        )  # add the outermost layer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Standard forward"""
        return self.model(input) # type: ignore[no-any-return]

class UNetSkipConnectionBlock(nn.Module):
    """
    Defines the UNet submodule with skip connection.
    """

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: Optional[int]=None,
        submodule: Optional[nn.Module]=None,
        outermost: bool=False,
        innermost: bool=False,
        norm_layer: Union[Type[nn.Module], functools.partial[nn.Module]]=nn.BatchNorm2d,
        use_dropout: bool=False,
    ) -> None:
        """
        Construct a UNet submodule with skip connections.

        :param outer_nc: the number of filters in the outer conv layer
        :param inner_nc: the number of filters in the inner conv layer
        :param input_nc: the number of channels in input images/features
        :param submodule: previously defined submodules
        :param outermost: if this module is the outermost module
        :param innermost: if this module is the innermost module
        :param norm_layer: normalization layer
        :param use_dropout: if use dropout layers.
        """
        super(UNetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        down: List[Optional[nn.Module]] = []
        up: List[Optional[nn.Module]] = []

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model) # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x) # type: ignore[no-any-return]
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)
