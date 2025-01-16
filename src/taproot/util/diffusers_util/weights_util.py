from ..weights_util import HostedWeights

__all__ = ["HostedLoRA", "HostedTextualInversion"]

class HostedLoRA(HostedWeights):
    recommended_scale: float = 1.0

class HostedTextualInversion(HostedWeights):
    pass
