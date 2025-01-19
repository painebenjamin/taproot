from taproot.pretrained import SigLIPSO400MVisionEncoder

class JoyCaptionAlphaV2VisionEncoder(SigLIPSO400MVisionEncoder):
    """
    JoyCaption's custom trained vision encoder.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/image-encoding-joy-caption-alpha-v2.safetensors"
