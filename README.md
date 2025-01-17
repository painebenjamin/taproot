<div align="center">
<img src="https://github.com/user-attachments/assets/f965fd42-2a95-4552-9b5f-465fc4037a91" width="650" /><br />
<em>An open source real-time AI inference engine for seamless scaling</em>
</div>
<hr/>
<p align="center">
    <img src="https://img.shields.io/static/v1?label=painebenjamin&message=taproot&color=234b0e&logo=github" alt="painebenjamin - taproot">
    <img src="https://img.shields.io/github/stars/painebenjamin/taproot?style=social" alt="stars - taproot">
    <img src="https://img.shields.io/github/forks/painebenjamin/taproot?style=social" alt="forks - taproot"><br />
    <a href="https://github.com/painebenjamin/taproot/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache-234b0e" alt="License"></a>
    <a href="https://pypi.org/project/taproot"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/taproot?color=234b0e"></a>
    <a href="https://pypistats.org/packages/taproot"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dd/taproot?logo=python&logoColor=white&color=234b0e"></a>
</p>

# About

Taproot is a seamlessly scalable AI/ML inference engine designed for deployment across hardware clusters with disparate capabilities.

## Why Taproot?

Most AI/ML inference engines are built for either large-scale cloud infrastructures or constrained edge devices - Taproot is designed for **medium-scale deployments**, offering flexible and distributed on-premise or PAYG setups. It efficiently uses older or consumer-grade hardware, making it suitable for small networks or ad-hoc clusters, without relying on centralized, hyperscale architectures.

Taproot is also **really, *really* fast** with latency as low as **50 microseconds** per request and transfer rates up to **2 GB/s** on consumer hardware, supporting standard HTTP/S, websockets, and raw TCP or Unix sockets.

<div align="center">
<a href="https://github.com/user-attachments/assets/c39ac7ad-d46b-4ab1-bd79-5bbd6595db52" target="_blank">
<img src="https://github.com/user-attachments/assets/c39ac7ad-d46b-4ab1-bd79-5bbd6595db52" width=640 />
</a><br />
<em>Taproot server/client round-trip echo times for varying packet sizes, grouped by supported protocol.</em>
</div><br/> 

Two encryption methods are also supported:
1. `tcps` uses raw `tcp` socket communication with bidirectional AES-NI encryption, configured with a key on server and client.
2. `wss` uses OpenSSL to serve standard TLS connections, configured with a key, certificate and optionally chain.

## Available Models

There are more than 170 models available across 18 task categories. See the [Task Catalog](https://github.com/painebenjamin/taproot/wiki/Task-Catalog) for the complete list, licenses, requirements and citations. Despite the large number of models available, there are many more yet to be added - if you're looking for a particular enhancement, don't hesitate to make an issue on this repository to request it.

### Roadmap

1. IP Adapter Models for Diffusers Image Generation Pipelines
2. ControlNet Models for Diffusers Image Generation Pipelines
3. Additional quantization backends for large models
    - Currently BitsandBytes (Int8/NF4) and GGUF (through llama.cpp) are supported with pre-quantized checkpoints available.
    - FP8 support through Optimum-Quanto, TorchAO and custom kernels is in development.
4. Improved multi-GPU support
    - This is currently supported through manual configuration, but usability can be improved.
5. Additional annotators/detectors for image and video
    - E.g. Marigold, SAM2
6. Additional audio generation models
    - E.g. Stable Audio, AudioLDM, MusicGen

# Installation

```sh
pip install taproot
```

Some additional packages are available to install with the square-bracket syntax (e.g. `pip install taproot[a,b,c]`), these are:
- **tools** - Additional packages for LLM tools like DuckDuckGo Search, BeautifulSoup (for web scraping), etc.
- **cli** - Additional packages for prettifying console output.
- **av** - Additional packages for reading and writing video.
- **uv** - `uvloop` for improved performance on linux systems.

## Installing Tasks

Some tasks are available immediately, but most tasks required additional packages and files. Install these tasks with `taproot install [task:model]+`, e.g: 

```sh
taproot install image-generation:stable-diffusion-xl
```

# Usage

## Command-Line

### Introspecting Tasks

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

### Invoking Tasks

Run `taproot invoke` to run any task from the command line. All parameters to the task can be passed as flags to the call using kebab-case, e.g.:

```sh
taproot invoke image-generation:stable-diffusion-xl \
    --prompt "a photograph of a golden retriever at the park" \
    --negative-prompt "fall, autumn, blurry, out-of-focus" \
    --seed 12345
Loading task.
100%|███████████████████████████████████████████████████████████████████████████| 7/7 [00:03<00:00,  2.27it/s]
Task loaded in 4.0 s.
Invoking task.
100%|█████████████████████████████████████████████████████████████████████████| 20/20 [00:04<00:00,  4.34it/s]
Task invoked in 6.5 s. Result:
8940aa12-66a7-4233-bfd6-f19da339b71b.png
```

## Python

### Direct Task Usage

```py
from taproot import Task
sdxl = Task.get("image-generation", "stable-diffusion-xl")
pipeline = sdxl()
pipeline.load() # Uses GPU 0 when available
pipeline(prompt="Hello, world!").save("./output.png")
```

### With a Remote Server

```py
import asyncio
from taproot import Tap

async def main() -> None:
    tap = Tap()
    tap.remote_address = "ws://127.0.0.1:32189"
    result = await tap("image-generation", model="stable-diffusion-xl", prompt="Hello, world!")
    result.save("./output.png")

asyncio.run(main)
```

### With a Local Server

Also shows usage with `uvloop`.

```py
import uvloop
from taproot import Tap

async def main() -> None:
    async with Tap.local() as tap:
        # Taproot is now running on ws://127.0.0.1:32189 with a local dispatcher
        result = await tap("speech-synthesis", model="kokoro", text="Hello, world!")
        result.save("./output.wav")

uvloop.run(main)
```

## Running Servers

Taproot uses a three-roled cluster structure:
1. **Overseers** are entry points into clusters, routing requests to one or more dispatchers.
2. **Dispatchers** are machines capable of running tasks by spawning executors.
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
