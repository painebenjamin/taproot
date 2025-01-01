from ..pretrained import SDXLVAE

__all__ = ["SDXLTurboVAE"]

class SDXLTurboVAE(SDXLVAE):
    """
    Enables compilation on the SDXLVAE.
    """
    use_compile: bool = True
