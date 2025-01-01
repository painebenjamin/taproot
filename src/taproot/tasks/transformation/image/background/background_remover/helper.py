# type: ignore
# adapted from https://github.com/nadermx/backgroundremover/
import os
import torch

from taproot.util import load_state_dict

class Net(torch.nn.Module):
    def __init__(
        self,
        path,
        model_name="u2net",
        device=torch.device("cpu"),
        dtype=None,
    ):
        super(Net, self).__init__()
        assert os.path.exists(path), "model path must exist"

        if model_name == "u2netp":
            from .u2net import U2NETP
            self.net = U2NETP(3, 1)
        elif model_name in ["u2net", "u2net_human_seg"]:
            from .u2net import U2NET
            self.net = U2NET(3, 1)
        else:
            raise ValueError("Choose between u2net, u2net_human_seg or u2netp")

        self.net.load_state_dict(load_state_dict(path))
        self.net.to(device=device, dtype=dtype)
        self.net.eval()

    def to(self, device: torch.device, dtype=None):
        self.net.to(device=device, dtype=dtype)
