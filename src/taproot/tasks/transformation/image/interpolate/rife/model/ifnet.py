# adapted from https://github.com/hzwer/Practical-RIFE/
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union

from .warper import Warper

__all__ = ["IFNet"]

class Head(nn.Module):
    """
    Head module for the IFNet
    """
    def __init__(
        self,
        num_channels: int=3,
        num_hidden_channels: int=16,
        out_channels: int=4,
        relu_slope: float=0.2,
    ) -> None:
        """
        :param num_channels: number of input channels
        :param num_hidden_channels: number of hidden channels
        :param out_channels: number of output channels
        :param relu_slope: slope of the LeakyReLU activation function
        """
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(num_channels, num_hidden_channels, 3, 2, 1)
        self.cnn1 = nn.Conv2d(num_hidden_channels, num_hidden_channels, 3, 1, 1)
        self.cnn2 = nn.Conv2d(num_hidden_channels, num_hidden_channels, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(num_hidden_channels, out_channels, 4, 2, 1)
        self.relu = nn.LeakyReLU(relu_slope, inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool=False
    ) -> Union[
        torch.Tensor,
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor
        ]
    ]:
        """
        :param x: input tensor
        :param return_features: whether to return intermediate features
        :return: output tensor or intermediate features and output tensor
        """
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)

        if return_features:
            return x0, x1, x2, x3

        return x3 # type: ignore[no-any-return]

class ResConv(nn.Module):
    """
    A residual convolutional block
    """
    def __init__(
        self,
        num_channels: int,
        dilation: int=1,
        relu_slope: float=0.2
    ) -> None:
        """
        :param num_channels: number of input and output channels
        :param dilation: dilation factor of the convolutional layer
        :param relu_slope: slope of the LeakyReLU activation function
        """
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        )
        self.beta = nn.Parameter(torch.ones((1, num_channels, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(relu_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor
        :return: output tensor
        """
        return self.relu(self.conv(x) * self.beta + x) # type: ignore[no-any-return]

class IFBlock(nn.Module):
    """
    An Image Flow block
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int=64,
        num_convs: int=8,
        last_conv_channels: int=52,
        relu_slope: float=0.2,
    ) -> None:
        """
        :param in_channels: number of input channels
        :param hidden_channels: number of hidden channels
        :param relu_slope: slope of the LeakyReLU activation function
        """
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels // 2, 3, 2, 1),
                nn.LeakyReLU(relu_slope, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(hidden_channels // 2, hidden_channels, 3, 2, 1),
                nn.LeakyReLU(relu_slope, inplace=True),
            )
        )
        self.convblock = nn.Sequential(*[
            ResConv(hidden_channels) for _ in range(num_convs)
        ])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, last_conv_channels, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(
        self,
        x: torch.Tensor,
        flow: Optional[torch.Tensor]=None,
        scale: int=1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input tensor
        :param flow: flow tensor
        :param scale: scaling factor
        :return: flow, mask, and feature tensors
        """
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat([x, flow], dim=1) # type: ignore[list-item]

        feat = self.conv0(x)
        feat = self.convblock(feat)

        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)

        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]

        return flow, mask, feat # type: ignore[return-value]

class IFNet(nn.Module):
    """
    An Image Flow network
    """
    def __init__(
        self,
        head_channels: int=3,
        head_hidden_channels: int=16,
        head_out_channels: int=4,
        block_channels: Tuple[int, ...] = (15, 28, 28, 28, 28),
        block_hidden_channels: Tuple[int, ...]=(192, 128, 96, 64, 32),
    ) -> None:
        super(IFNet, self).__init__()
        self.warper = Warper()
        self.encode = Head(
            num_channels=head_channels,
            num_hidden_channels=head_hidden_channels,
            out_channels=head_out_channels
        )

        assert len(block_channels) == len(block_hidden_channels), "The number of block channels and hidden channels must be the same"

        self.num_blocks = len(block_channels)

        for i, (block_channel, block_hidden_channel) in enumerate(zip(block_channels, block_hidden_channels)):
            block = IFBlock(in_channels=block_channel, hidden_channels=block_hidden_channel)
            self.add_module(f"block{i}", block)

    @property
    def block_list(self) -> List[IFBlock]:
        """
        :return: list of IFBlocks
        """
        return [getattr(self, f"block{i}") for i in range(self.num_blocks)]

    def forward(
        self,
        x: torch.Tensor,
        timestep: Union[float, torch.Tensor]=0.5,
        scale_list: Tuple[int, ...]=(16, 8, 4, 2, 1),
    ) -> torch.Tensor:
        """
        :param x: input tensor
        :param timestep: time step
        :param scale_list: list of scaling factors
        :return: output tensor
        """
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]

        if not torch.is_tensor(timestep): # type: ignore[no-untyped-call]
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3]) # type: ignore[union-attr]

        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])

        warped_img0 = img0
        warped_img1 = img1

        flow: Optional[torch.Tensor] = None
        mask: Optional[torch.Tensor] = None

        assert len(scale_list) == self.num_blocks, "The number of scales must be the same as the number of blocks"

        for block, scale in zip(self.block_list, scale_list):
            if flow is None or mask is None:
                flow, mask, feat = block(
                    torch.cat([
                        img0[:, :3],
                        img1[:, :3],
                        f0,
                        f1,
                        timestep
                    ], dim=1),
                    flow=None,
                    scale=scale
                )
            else:
                wf0 = self.warper(f0, flow[:, :2])
                wf1 = self.warper(f1, flow[:, 2:4])
                flow_d, mask_d, feat_d = block(
                    torch.cat([
                        warped_img0[:, :3],
                        warped_img1[:, :3],
                        wf0,
                        wf1,
                        timestep,
                        mask,
                        feat
                    ], dim=1),
                    flow=flow,
                    scale=scale
                )
                mask = mask_d
                flow = flow + flow_d

            warped_img0 = self.warper(img0, flow[:, :2]) # type: ignore[index]
            warped_img1 = self.warper(img1, flow[:, 2:4]) # type: ignore[index]

        mask = torch.sigmoid(mask) # type: ignore[arg-type]
        return (warped_img0 * mask + warped_img1 * (1 - mask)) # type: ignore[no-any-return]
