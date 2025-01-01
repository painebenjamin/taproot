# type: ignore
# adapted from https://github.com/nadermx/backgroundremover/
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .data_loader import (
    RescaleT,
    ToTensorLab
)

def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose(
        [RescaleT(320), ToTensorLab(flag=0)]
    )
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})
    return sample

def predict(net, item, device):
    with torch.no_grad():
        samples = torch.stack([
            torch.Tensor(
                preprocess(
                    np.array(i.convert("RGB"))
                )["image"].float()
            )
            for i in item
        ]).to(device)
        d1 = net(samples)[0]
        pred = norm_pred(d1[:, 0, :, :])
        images = [
            Image.fromarray(
                p.cpu().detach().numpy() * 255
            ).convert("L")
            for p in pred
        ]
        del pred, d1, samples
        return images
