# Adapted from https://github.com/Zyphra/Zonos
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio # type: ignore[import-untyped]

from typing import Type, Any, List, Tuple
from typing_extensions import Literal

__all__ = [
    "FeatureExtractor",
    "ASP",
    "BasicBlock",
    "SimpleAttentionBasicBlock",
    "Bottleneck",
    "ResNet",
    "ResNet293",
    "SpeakerEmbedding",
    "SpeakerEmbeddingLDA",
]

class FeatureExtractor(nn.Module):
    """
    Feature Extractor
    """
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 512,
        win_length: float = 0.025,
        hop_length: float = 0.01,
        n_mels: int = 80,
    ) -> None:
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=int(win_length * sample_rate),
            hop_length=int(hop_length * sample_rate),
            n_mels=n_mels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spec(x)
        out = torch.log(out + 1e-6)
        out = out - out.mean(axis=2).unsqueeze(dim=2)
        return out # type: ignore[no-any-return]

class ASP(nn.Module):
    """
    Attention Statistics Pooling
    """
    def __init__(
        self,
        in_planes: int,
        acoustic_dim: int,
    ) -> None:
        super(ASP, self).__init__()
        outmap_size = int(acoustic_dim / 8)
        self.out_dim = in_planes * 8 * outmap_size * 2
        self.attention = nn.Sequential(
            nn.Conv1d(in_planes * 8 * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, in_planes * 8 * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size()[0], -1, x.size()[-1])
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        x = torch.cat((mu, sg), 1)
        x = x.view(x.size()[0], -1)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        conv_layer_cls: Type[nn.Module],
        norm_layer_cls: Type[nn.Module],
        in_planes: int,
        planes: int,
        stride: int=1,
        block_id: int=1
    ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer_cls(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer_cls(planes)
        self.conv2 = conv_layer_cls(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer_cls(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                conv_layer_cls(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                norm_layer_cls(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out # type: ignore[no-any-return]

class SimpleAttentionBasicBlock(BasicBlock):
    """
    Simple Attention Basic Block
    """
    expansion = 1

    def __init__(
        self,
        conv_layer_cls: Type[nn.Module],
        norm_layer_cls: Type[nn.Module],
        in_planes: int,
        planes: int,
        stride: int=1,
        block_id: int=1
    ) -> None:
        super(SimpleAttentionBasicBlock, self).__init__(
            conv_layer_cls=conv_layer_cls,
            norm_layer_cls=norm_layer_cls,
            in_planes=in_planes,
            planes=planes,
            stride=stride,
            block_id=block_id
        )
        self.sigmoid = nn.Sigmoid()
        self.bn3 = norm_layer_cls(planes * self.expansion)

    def attention(self, x: torch.Tensor, lambda_p: float=1e-4) -> torch.Tensor:
        n = x.shape[2] * x.shape[3] - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        e_inv = d / (4 * (v + lambda_p)) + 0.5
        return x * self.sigmoid(e_inv) # type: ignore[no-any-return]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        out += self.downsample(x)
        out = self.relu(out)
        return out # type: ignore[no-any-return]

class Bottleneck(BasicBlock):
    expansion = 4

    def __init__(
        self,
        conv_layer_cls: Type[nn.Module],
        norm_layer_cls: Type[nn.Module],
        in_planes: int,
        planes: int,
        stride: int=1,
        block_id: int=1
    ) -> None:
        super(Bottleneck, self).__init__(
            conv_layer_cls=conv_layer_cls,
            norm_layer_cls=norm_layer_cls,
            in_planes=in_planes,
            planes=planes,
            stride=stride,
            block_id=block_id
        )
        self.shortcut = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out)) # type: ignore[operator]
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """
    Residual Network
    """
    def __init__(
        self,
        in_planes: int,
        num_blocks: List[int],
        block_cls: Type[BasicBlock]=BasicBlock,
        in_ch: int=1,
        feat_dim: Literal["1d", "2d", "3d"]="2d",
        **kwargs: Any
    ) -> None:
        super(ResNet, self).__init__()
        norm_layer_cls: Type[nn.Module] = nn.Module
        conv_layer_cls: Type[nn.Module] = nn.Module
        if feat_dim == "1d":
            norm_layer_cls = nn.BatchNorm1d
            conv_layer_cls = nn.Conv1d
        elif feat_dim == "2d":
            norm_layer_cls = nn.BatchNorm2d
            conv_layer_cls = nn.Conv2d
        elif feat_dim == "3d":
            norm_layer_cls = nn.BatchNorm3d
            conv_layer_cls = nn.Conv3d
        else:
            raise ValueError("Invalid feat_dim, expected '1d', '2d', or '3d'")

        self.conv1 = conv_layer_cls(in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer_cls(in_planes)
        self.relu = nn.ReLU(inplace=True)

        num_planes = [in_planes * 2**i for i in range(len(num_blocks))]

        def make_layer(
            planes: int,
            blocks: int,
            stride: int,
            block_id: int=1
        ) -> nn.Module:
            nonlocal in_planes
            strides = [stride] + [1] * (blocks - 1)
            layers = []
            for stride in strides:
                layers.append(
                    block_cls(
                        conv_layer_cls,
                        norm_layer_cls,
                        in_planes,
                        planes,
                        stride,
                        block_id
                    )
                )
                in_planes = planes * layers[-1].expansion
            return nn.Sequential(*layers)

        self.layers = nn.Sequential(*[
            make_layer(
                planes=planes,
                blocks=blocks,
                stride=max(min(i+1, 2), 1),
                block_id=i+1
            )
            for i, (planes, blocks) in enumerate(zip(num_planes, num_blocks))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        return x

class ResNet293(ResNet):
    def __init__(
        self,
        in_planes: int,
        **kwargs: Any
    ) -> None:
        super(ResNet293, self).__init__(
            in_planes,
            num_blocks=[10, 20, 64, 3],
            block_cls=SimpleAttentionBasicBlock,
            **kwargs
        )

class SpeakerEmbedding(nn.Module):
    def __init__(
        self,
        in_planes: int = 64,
        embedding_dim: int = 256,
        acoustic_dim: int = 80,
        dropout: float = 0,
        **kwargs: Any
    ) -> None:
        super(SpeakerEmbedding, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.front = ResNet293(in_planes)
        self.pooling = ASP(in_planes, acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embedding_dim)
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.feature_extractor.to(x.device)
        x = self.feature_extractor(x)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        return x

class SpeakerEmbeddingLDA(SpeakerEmbedding):
    def __init__(
        self,
        in_planes: int = 64,
        embedding_dim: int = 256,
        out_dim: int = 128,
        acoustic_dim: int = 80,
        dropout: float = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(
            in_planes=in_planes,
            embedding_dim=embedding_dim,
            acoustic_dim=acoustic_dim,
            dropout=dropout,
            **kwargs
        )
        self.lda = nn.Linear(
            embedding_dim,
            out_dim,
            bias=True,
        )

    def forward( # type: ignore[override]
        self,
        wav: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = super().forward(wav)
        return emb, self.lda(emb)
