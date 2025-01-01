from typing_extensions import Literal

__all__ = [
    "TEST_RESTRICTED_MODULES",
    "KNOWN_GPU_MODEL_FILE_EXTENSIONS",
    "CONTROLNET_TYPE_LITERAL",
    "DIFFUSERS_SCHEDULER_LITERAL",
    "DIFFUSERS_MODEL_TYPE_LITERAL",
    "MULTIDIFFUSION_MASK_TYPE_LITERAL",
    "IMAGE_OUTPUT_FORMAT_LITERAL",
    "VIDEO_OUTPUT_FORMAT_LITERAL",
    "AUDIO_OUTPUT_FORMAT_LITERAL",
    "PROTOCOL_LITERAL",
    "SCHEME_LITERAL",
    "IMAGE_FIT_LITERAL",
    "IMAGE_ANCHOR_LITERAL",
    "IMAGE_SPACE_LITERAL",
    "IMAGE_CAPTION_COLOR_LITERAL",
    "IMAGE_CAPTION_TYPE_LITERAL",
    "IMAGE_CAPTION_ALIGN_LITERAL",
    "IMAGE_CAPTION_POSITION_LITERAL",
]

TEST_RESTRICTED_MODULES = {
    "accelerate", "cv2", "deepfilternet", "diffusers",
    "einops", "hydra", "librosa","mmaction", "mmcls",
    "mmcv", "mmdet", "mmengine", "mmocr", "mmpose",
    "mmseg", "mmtracking", "numpy", "onnx", "onnxruntime",
    "pandas", "peft", "pil", "pytorch_lightning", "safetensors",
    "scipy", "skimage", "sklearn", "timm", "torch", "torchaudio",
    "torchdiffeq", "torchvision", "transformers", "tts",
    "xformers",
}
KNOWN_GPU_MODEL_FILE_EXTENSIONS = {
    ".bin", ".pt", ".pth", ".onnx", ".h5", ".gguf",
    ".pb", ".pbtxt", ".tflite", ".uff", ".engine",
    ".trt", ".mlmodel", ".ckpt", ".model", ".safetensors",
}
CONTROLNET_TYPE_LITERAL = Literal[
    "anime-line-art", "canny", "depth",
    "hed", "line-art", "normal", "pose",
    "qr", "scribble", "soft-edge", "tile"
]
DIFFUSERS_SCHEDULER_LITERAL = Literal[
    "ddim", "ddpm", "ddpm_wuerstchen", "deis_multistep",
    "dpm_cogvideox", "dpmsolver_multistep", "dpmsolver_multistep_karras",
    "dpmsolver_sde", "dpmsolver_sde", "dpmsolver_sde_multistep",
    "dpmsolver_sde_multistep_karras", "dpmsolver_singlestep",
    "dpmsolver_singlestep_karras", "edm_dpmsolver_multistep", "edm_euler",
    "euler_ancestral_discrete", "euler_discrete", "euler_discrete_karras",
    "flow_match_euler_discrete", "flow_match_heun_discrete", "heun_discrete",
    "ipndm", "k_dpm_2_ancestral_discrete", "k_dpm_2_ancestral_discrete_karras",
    "k_dpm_2_discrete", "k_dpm_2_discrete_karras", "lcm", "lms_discrete",
    "lms_discrete_karras", "pndm", "tcd", "unipc"
]
DIFFUSERS_MODEL_TYPE_LITERAL = Literal["sd", "sdxl", "sd3", "flux"]
MULTIDIFFUSION_MASK_TYPE_LITERAL = Literal["gaussian", "bilinear", "constant"]
PROTOCOL_LITERAL = Literal["memory", "unix", "tcp", "ws"]
SCHEME_LITERAL = Literal["memory", "unix", "tcp", "tcps", "ws", "wss"]
IMAGE_OUTPUT_FORMAT_LITERAL = Literal["png", "jpeg", "float", "int", "latent"]
VIDEO_OUTPUT_FORMAT_LITERAL = Literal["mp4", "gif", "png", "float", "int", "latent"]
AUDIO_OUTPUT_FORMAT_LITERAL = Literal["wav", "mp3", "ogg", "flac", "float", "int"]
IMAGE_FIT_LITERAL = Literal["actual", "stretch", "cover", "contain"]
IMAGE_ANCHOR_LITERAL = Literal[
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center-center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
]
IMAGE_SPACE_LITERAL = Literal["latent", "pixel"]
IMAGE_CAPTION_COLOR_LITERAL = Literal["black-on-white", "white-on-black"]
IMAGE_CAPTION_TYPE_LITERAL = Literal["overlay", "separate"]
IMAGE_CAPTION_ALIGN_LITERAL = Literal["left", "center", "right"]
IMAGE_CAPTION_POSITION_LITERAL = Literal["top", "center", "bottom"]
