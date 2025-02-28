from typing import Optional
from taproot.constants import *
from ..base import LlamaTextGeneration

__all__ = [
    "CosmosPromptUpsamplerTextGenerationQ80",
    "CosmosPromptUpsamplerTextGenerationQ6K",
    "CosmosPromptUpsamplerTextGenerationQ5KM",
    "CosmosPromptUpsamplerTextGenerationQ4KM",
    "CosmosPromptUpsamplerTextGenerationQ3KM",
]

class CosmosPromptUpsamplerTextGeneration(LlamaTextGeneration):
    """
    Text generation using llama.cpp and NVIDIA's Cosmos 1.0 Prompt Upsampler LLM (based on Mistral NeMo).
    """
    """Global Task Metadata"""
    chat_format: Optional[str] = None

    """Authorship Metadata"""
    author: Optional[str] = "NVIDIA"

    """License Metadata"""
    license: Optional[str] = LICENSE_NVIDIA_OPEN_MODEL

class CosmosPromptUpsamplerTextGenerationQ80(CosmosPromptUpsamplerTextGeneration):
    task = "text-generation"
    model = "cosmos-prompt-upsampler-q8-0"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-cosmos-v1-0-prompt-upsampler-q8-0.gguf"
    display_name = "Cosmos Prompt Upsampler (Q8-0)"
    max_context_length = 131072

class CosmosPromptUpsamplerTextGenerationQ6K(CosmosPromptUpsamplerTextGeneration):
    task = "text-generation"
    model = "cosmos-prompt-upsampler-q6-k"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-cosmos-v1-0-prompt-upsampler-q6-k.gguf"
    display_name = "Cosmos Prompt Upsampler (Q6-K)"
    max_context_length = 131072

class CosmosPromptUpsamplerTextGenerationQ5KM(CosmosPromptUpsamplerTextGeneration):
    task = "text-generation"
    model = "cosmos-prompt-upsampler-q5-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-cosmos-v1-0-prompt-upsampler-q5-k-m.gguf"
    display_name = "Cosmos Prompt Upsampler (Q5-K-M)"
    max_context_length = 131072

class CosmosPromptUpsamplerTextGenerationQ4KM(CosmosPromptUpsamplerTextGeneration):
    task = "text-generation"
    model = "cosmos-prompt-upsampler-q4-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-cosmos-v1-0-prompt-upsampler-q4-k-m.gguf"
    display_name = "Cosmos Prompt Upsampler (Q4-K-M)"
    max_context_length = 131072

class CosmosPromptUpsamplerTextGenerationQ3KM(CosmosPromptUpsamplerTextGeneration):
    task = "text-generation"
    model = "cosmos-prompt-upsampler-q3-k-m"
    default = False
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-generation-cosmos-v1-0-prompt-upsampler-q3-k-m.gguf"
    display_name = "Cosmos Prompt Upsampler (Q3-K-M)"
    max_context_length = 131072
