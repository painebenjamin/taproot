<div align="center">
<img src="https://github.com/user-attachments/assets/f965fd42-2a95-4552-9b5f-465fc4037a91" /><br />
<em><strong>TAPROOT</strong> - an open source real-time AI inference engine with seamless scaling</em>
</div>

# This code will be released soon!

# Installation

```sh
pip install taproot
```

## Installing Tasks

Some tasks are available immediately, but most tasks required additional packages and files. Install these tasks with `taproot install [task:model]+`, e.g: 

```sh
taproot install image-generation:stable-diffusion-xl
```

# Usage

## Introspecting Tasks

From the command line, execute `taproot tasks` to see all tasks and their availability status, or `taproot info` for individual task information. For example:

```sh
taproot info image-generation stable-diffusion-xl

Stable Diffusion XL Image Generation (image-generation:stable-diffusion-xl, available)
    Generate an image from text and/or images using a stable diffusion XL model.
Hardware Requirements:                  
    GPU Required for Optimal Performance                                           
    Floating Point Precision: half                                                 
    Minimum Memory (CPU RAM) Required: 231.71 MB     
    Minimum Memory (GPU VRAM) Required: 7.58 GB               
Author:                          
    Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna and Robin Rombach
    Published in arXiv, vol. 2307.01952, “SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis”, 2023
    https://arxiv.org/abs/2307.01952                                               
License:
    OpenRAIL++-M License (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
    ✅ Attribution Required
    ✅ Derivatives Allowed
    ✅ Redistribution Allowed
    ✅ Copyleft (Share-Alike) Required
    ✅ Commercial Use Allowed
    ✅ Hosting Allowed
Files:                                                                             
    image-generation-stable-diffusion-xl-base-vae.fp16.safetensors (334.64 MB) [downloaded]
    image-generation-stable-diffusion-xl-base-unet.fp16.safetensors (5.14 GB) [downloaded]
    text-encoding-clip-vit-l.bf16.safetensors (246.14 MB) [downloaded]
    text-encoding-open-clip-vit-g.fp16.safetensors (1.39 GB) [downloaded]
    text-encoding-clip-vit-l-tokenizer-vocab.json (1.06 MB) [downloaded]
    text-encoding-clip-vit-l-tokenizer-special-tokens-map.json (588.00 B) [downloaded]
    text-encoding-clip-vit-l-tokenizer-merges.txt (524.62 KB) [downloaded]
    text-encoding-open-clip-vit-g-tokenizer-vocab.json (1.06 MB) [downloaded]
    text-encoding-open-clip-vit-g-tokenizer-special-tokens-map.json (576.00 B) [downloaded]
    text-encoding-open-clip-vit-g-tokenizer-merges.txt (524.62 KB) [downloaded]
    Total File Size: 7.11 GB
Required packages:
    pil~=9.5 [installed]
    torch<2.5,>=2.4 [installed]
    numpy~=1.22 [installed]
    diffusers>=0.29 [installed]
    torchvision<0.20,>=0.19 [installed]
    transformers>=4.41 [installed]
    safetensors~=0.4 [installed]
    accelerate~=1.0 [installed]
    sentencepiece~=0.2 [installed]
    compel~=2.0 [installed]
    peft~=0.13 [installed]
Signature:
    prompt: Union[str, List[str]], required
    prompt_2: Union[str, List[str]], default: None
    negative_prompt: Union[str, List[str]], default: None
    negative_prompt_2: Union[str, List[str]], default: None
    image: ImageType, default: None
    mask_image: ImageType, default: None
    guidance_scale: float, default: 5.0
    guidance_rescale: float, default: 0.0
    num_inference_steps: int, default: 20
    num_images_per_prompt: int, default: 1
    height: int, default: None
    width: int, default: None
    timesteps: List[int], default: None
    sigmas: List[float], default: None
    denoising_end: float, default: None
    strength: float, default: None
    latents: torch.Tensor, default: None
    prompt_embeds: torch.Tensor, default: None
    negative_prompt_embeds: torch.Tensor, default: None
    pooled_prompt_embeds: torch.Tensor, default: None
    negative_pooled_prompt_embeds: torch.Tensor, default: None
    clip_skip: int, default: None
    seed: SeedType, default: None
    pag_scale: float, default: None
    pag_adaptive_scale: float, default: None
    scheduler: Literal[ddim, ddpm, ddpm_wuerstchen, deis_multistep, dpm_cogvideox, dpmsolver_multistep, dpmsolver_multistep_karras, dpmsolver_sde, dpmsolver_sde_multistep, dpmsolver_sde_multistep_karras, dpmsolver_singlestep, dpmsolver_singlestep_karras, edm_dpmsolver_multistep, edm_euler, euler_ancestral_discrete, euler_discrete, euler_discrete_karras, flow_match_euler_discrete, flow_match_heun_discrete, heun_discrete, ipndm, k_dpm_2_ancestral_discrete, k_dpm_2_ancestral_discrete_karras, k_dpm_2_discrete, k_dpm_2_discrete_karras, lcm, lms_discrete, lms_discrete_karras, pndm, tcd, unipc], default: None
    output_format: Literal[png, jpeg, float, int, latent], default: png
    output_upload: bool, default: False
    highres_fix_factor: float, default: 1.0
    highres_fix_strength: float, default: None
    spatial_prompts: SpatialPromptInputType, default: None
Returns:
    ImageResultType
```

## Invoking Tasks

### Direct Task Usage

```py
from taproot import Task
sdxl = Task.get("image-generation", "stable-diffusion-xl")
pipeline = sdxl()
pipeline.load()
pipeline(prompt="Hello, world!").save("./output.png")
```

### With a Remote Server

```py
from taproot import Tap
tap = Tap()
tap.remote_address = "ws://127.0.0.1:32189"
result = tap.call("image-generation", model="stable-diffusion-xl", prompt="Hello, world!")
result.save("./output.png")
```

### With a Local Server

Also shows asynchronous usage.

```py
import asyncio
from taproot import Tap
tap = Tap()
with tap.local() as tap:
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(tap("image-generation", model="stable-diffusion-xl", prompt="Hello, world!"))
    result.save("./output.png")
```

## Running Servers

Taproot uses a three-roled cluster structure:
1. **Overseers** are entry points into clusters, routing requests to one or more dispatchers.
2. **Dispatchers** are machines capable of executing tasks by spawning executors.
3. **Executors** are servers ready to execute a task.

The simplest way to run a server is to run an overseer simultaneously with a local dispatcher like so:

```sh
taproot overseer --local
```

This will run on the default address of `ws://127.0.0.1:32189`, suitable for interaction from python or the browser.

There are many deployment possibilities across networks, with configuration available for encryption, listening addresses, and more. See the wiki for details (coming soon.)

## Outside Python

- [taproot.js](https://github.com/painebenjamin/taproot.js) - for the browser and node.js, available in ESM, UMD and IIFE
- taproot.php - coming soon
