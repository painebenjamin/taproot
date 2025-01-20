from ..weights_util import PretrainedWeights

__all__ = ["PretrainedLoRA", "PretrainedTextualInversion"]

class PretrainedLoRA(PretrainedWeights):
    recommended_scale: float = 1.0

class PretrainedTextualInversion(PretrainedWeights):
    pass
