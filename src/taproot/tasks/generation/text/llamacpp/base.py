from __future__ import annotations

import re
import json
import tempfile

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    TYPE_CHECKING
)
from typing_extensions import TypedDict, NotRequired

from taproot.util import (
    logger,
    to_pil_array,
    serialize_image,
    get_seed,
    catch_output,
    to_pil_array,
    image_tiles,
    maybe_use_tqdm,
    sliding_window_count
)
from taproot.constants import *
from taproot.tasks.base import Task

from ..tools import (
    Tool,
    LlamaToolMetadata,
    LlamaToolFunctionCall,
    LlamaToolFunctionCallResult
)

if TYPE_CHECKING:
    from taproot.hinting import PromptType, MessageDict, SeedType, ImageType
    from llama_cpp import Llama # type: ignore[import-not-found,unused-ignore]
    from llama_cpp.llama_tokenizer import LlamaHFTokenizer # type: ignore[import-not-found,unused-ignore]
    from ..roles import Role

class LlamaImageURLDict(TypedDict):
    """
    The image URL of a message to the Llama C++ library.
    """
    url: str

class LlamaContentDict(TypedDict):
    """
    The content of a message to the Llama C++ library.
    """
    type: Literal["text", "image_url"]
    text: NotRequired[str]
    image_url: NotRequired[LlamaImageURLDict]

class LlamaMessageDict(TypedDict):
    """
    A dictionary representing a message to the Llama C++ library.
    """
    role: Literal["user", "system", "assistant"]
    content: Union[str, List[LlamaContentDict]]

__all__ = [
    "LlamaTextGeneration",
    "LlamaImageCaptioning"
]

class LlamaTextGeneration(Task):
    """
    A base class for tasks that use the Llama C++ library.
    """
    # Local properties
    llama: Llama
    roles: Dict[str, Role] = {}

    # Configurables
    model_url: Optional[str] = None
    mmproj_url: Optional[str] = None
    chat_format: Optional[str] = None
    default_role: Optional[str] = None
    supports_image: bool = False
    response_start_text: Optional[str] = None
    response_end_text: Optional[Union[str, List[str]]] = ["INST", "ASSISTANT", "USER"]
    max_context_length: Optional[int] = None # When set, calculate context length based on available memory
    vocab_length: Optional[int] = 128256
    bytes_per_cache_token: Optional[int] = 4

    # Overrides
    gpu_precision: Optional[str] = "half" # Quantized models run as half precision
    measure_nvidia_smi: bool = True # Require external tool to measure

    # Internal typing
    _tool_metadata: List[LlamaToolMetadata]

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Required packages for the task.
        """
        return {
            "llama_cpp": LLAMA_CPP_VERSION_SPEC,
            "torch": TORCH_VERSION_SPEC,
            "numpy": NUMPY_VERSION_SPEC,
            "pil": PILLOW_VERSION_SPEC
        }

    @classmethod
    def required_files(cls, allow_optional: bool=True) -> List[str]:
        """
        Required files
        """
        if cls.model_url is None:
            return []
        urls = [cls.model_url]
        if cls.mmproj_url is not None:
            urls.append(cls.mmproj_url)
        return urls

    """Overrideable methods"""

    def get_tokenizer(cls) -> Optional[LlamaHFTokenizer]:
        """
        Get the tokenizer for the task
        """
        return None

    """Internal properties for task"""

    @property
    def model_file(self) -> str:
        """
        Model file
        """
        if self.model_url is None:
            raise ValueError(f"Model URL is not set for class {self.__class__.__name__}")
        return self.get_model_file(self.model_url)

    @property
    def intermediate_text(self) -> str:
        """
        Intermediates as text without parsing
        """
        return "".join(self.intermediates)

    @property
    def tools(self) -> Dict[str, Type[Tool]]:
        """
        Tools to use for the task
        """
        if not hasattr(self, "_tools"):
            self._tools = Tool.enumerate()
        return self._tools

    @property
    def tool_metadata(self) -> List[LlamaToolMetadata]:
        """
        Tools metadata
        """
        if not hasattr(self, "_tool_metadata"):
            self._tool_metadata = []
            for tool_class in self.tools.values():
                self._tool_metadata.append(tool_class.llama_metadata())
        return self._tool_metadata

    """Override properties"""

    @property
    def last_intermediate(self) -> Any:
        """
        Override this to concatenate intermediates, instead of returning the last one.
        """
        return self.trim_response(self.intermediate_text)

    """Public Methods"""

    def trim_response(self, text: str) -> str:
        """
        Trims a response to remove any unnecessary text.
        """
        text = text.strip(" \r\n\t'\"")
        if self.response_start_text is not None:
            start_index = text.find(self.response_start_text)
            if start_index != -1:
                text = text[start_index + len(self.response_start_text):]
        # If the text is diarized, remove the diarization
        text = re.sub(r"\[.*?\]", "", text)
        text = text.rstrip("([{")
        text = text.lstrip(")]}")
        text = text.strip(" \r\n\t'\"\\/")
        return text

    def get_role(self, role_name: Optional[str]=None) -> Role:
        """
        Get the role for the task
        """
        from ..roles import Role
        if role_name is None:
            role_name = self.default_role
        if not role_name:
            return Role()
        if role_name not in self.roles:
            self.roles[role_name] = Role.get(role_name)()
        return self.roles[role_name]

    def format_conversation(self, conversation: List[MessageDict]) -> List[LlamaMessageDict]:
        """
        Format the conversation for the Llama C++ library
        """
        formatted_conversation: List[LlamaMessageDict] = []
        for message in conversation:
            message_image = message.get("image", None)
            if message_image:
                if isinstance(message_image, str) and message_image.startswith("http"):
                    image_url = message_image
                else:
                    image_object = to_pil_array(message_image)[0]
                    image_url = serialize_image(image_object)
            else:
                image_url = None
            if image_url:
                formatted_conversation.append({
                    "role": message["role"],
                    "content": [
                        {"type": "text", "text": message["text"]},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                })
            else:
                formatted_conversation.append({
                    "role": message["role"],
                    "content": message["text"]
                })
        return formatted_conversation

    def get_stop_tokens(
        self,
        stop: Optional[Union[str, List[str]]]=None,
    ) -> List[str]:
        """
        Get the stop tokens for the task
        """
        # Merge stop tokens
        stop_tokens: List[str] = []
        if stop is not None:
            if isinstance(stop, list):
                stop_tokens.extend(stop)
            else:
                stop_tokens.append(stop)
        if self.response_end_text is not None:
            if isinstance(self.response_end_text, list):
                stop_tokens.extend(self.response_end_text)
            else:
                stop_tokens.append(self.response_end_text)
        # Remove any empty or none tokens
        stop_tokens = [token for token in stop_tokens if token]
        return stop_tokens

    def execute_function(
        self,
        function_call: LlamaToolFunctionCall
    ) -> LlamaToolFunctionCallResult:
        """
        Given a Llama-formatted function call, executes the function
        """
        tool_name = function_call["name"]

        tool_class = self.tools.get(tool_name, None)
        assert tool_class is not None, f"Tool {tool_name} not found!"

        tool_parameters: Dict[str, Any] = {}
        arguments = function_call.get("arguments", None)
        if isinstance(arguments, str):
            try:
                tool_parameters = json.loads(arguments)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON arguments for tool {tool_name}: {arguments}")
            assert isinstance(tool_parameters, dict), "Tool parameters must be a dictionary"

        tool_instance = tool_class()

        try:
            result = tool_instance(**tool_parameters)
        except Exception as e:
            result = f"Error: {type(e).__name__}: {str(e)}"

        return {
            "function": function_call,
            "result": result,
            "citations": tool_instance.citations
        }

    def execute_and_format_function(
        self,
        prompt: PromptType,
        function_call: LlamaToolFunctionCall,
        history: Optional[PromptType]=None,
        seed: SeedType=None,
        max_tokens: Optional[int]=None,
        stop: Optional[Union[str, List[str]]]=None,
        temperature: float=0.2,
        top_p: float=0.95,
        top_k: int=40,
        min_p: float=0.5,
        typical_p: float=1.0,
        presence_penalty: float=0.0,
        frequency_penalty: float=0.0,
        repeat_penalty: float=1.1,
        stream: bool=False,
    ) -> LlamaToolFunctionCallResult:
        """
        Executes a function call and formats the result
        """
        function_name = function_call["name"]
        function_result = self.execute_function(function_call)
        function_result["formatted"] = self.format_function_result(
            prompt,
            function_name,
            function_result["result"],
            history=history,
            seed=get_seed(seed),
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            stream=stream,
        )
        return function_result

    def format_function_result(
        self,
        prompt: PromptType,
        function_name: str,
        result: Any,
        history: Optional[PromptType]=None,
        seed: SeedType=None,
        max_tokens: Optional[int]=None,
        stop: Optional[Union[str, List[str]]]=None,
        temperature: float=0.2,
        top_p: float=0.95,
        top_k: int=40,
        min_p: float=0.5,
        typical_p: float=1.0,
        presence_penalty: float=0.0,
        frequency_penalty: float=0.0,
        repeat_penalty: float=1.1,
        stream: bool=False,
    ) -> str:
        """
        Formats the result of a function call using the `tool-result-formatter` role.
        """
        conversation = self.get_role("tool-result-formatter").get_conversation(
            prompt,
            history=history,
            tool=function_name,
            result=result
        )
        logger.debug(f"Formatting tool result with conversation: {conversation}")
        return self.llama.create_chat_completion( # type: ignore[no-any-return,return-value,unused-ignore]
            messages=self.format_conversation(conversation), # type: ignore[arg-type,unused-ignore]
            seed=get_seed(seed),
            max_tokens=max_tokens,
            stop=self.get_stop_tokens(stop),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            stream=stream,
        )

    def get_tool(
        self,
        prompt: PromptType,
        history: Optional[PromptType]=None,
        seed: SeedType=None,
        tools: Optional[List[LlamaToolMetadata]]=None,
        temperature: float=0.0,
        top_p: float=0.98,
        top_k: int=10,
        min_p: float=0.6,
    ) -> Optional[str]:
        """
        Given a prompt, queries for the tool to use.
        This is a chain-of-through patch for models that don't support 'auto' tool choice.
        """
        tools_list = tools or self.tool_metadata
        tools_dict = {
            tool["function"]["name"]: tool["function"]["description"]
            for tool in tools_list
        }
        conversation = self.get_role("tool-picker").get_conversation(
            prompt,
            conversation_history=history,
            available_tools=tools_dict
        )
        logger.debug(f"Giving tool picker prompt: {conversation}")
        generation = self.llama.create_chat_completion(
            messages=self.format_conversation(conversation), # type: ignore[arg-type,unused-ignore]
            seed=get_seed(seed),
            max_tokens=4,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        result = generation["choices"][0]["message"].get("content", "").strip().lower() # type: ignore[index,union-attr,unused-ignore]
        if result == "none" or result not in tools_dict:
            logger.debug(f"Result {result} not a tool, returning None.")
            return None
        logger.debug(f"Chose tool: {result}")
        return result # type: ignore[no-any-return,unused-ignore]

    def get_context_length(self) -> int:
        """
        Context length for the model.
        Defaults to 0, which reads the model's config.
        """
        if self.max_context_length is not None and self.bytes_per_cache_token is not None and self.vocab_length is not None:
            static_gpu_mem = self.required_static_gpu_memory_gb()
            if static_gpu_mem is None:
                static_gpu_mem = 0
            static_bytes = static_gpu_mem * 1024 * 1024 * 1024
            free_bytes = self.gpu.memory_free 
            # Each additional token of context requires P * V bytes of memory,
            # where P is the number of bytes to store a token and V is the vocab size.
            return min(
                self.max_context_length,
                int(free_bytes // (self.bytes_per_cache_token * self.vocab_length))
            )
        return 0

    def get_chat_handler(self) -> Any:
        """
        Gets the llama CPP chat handler. Default returns none.
        """
        return None

    """Override methods"""

    def load(self) -> None:
        """
        Load the model
        """
        import logging
        from llama_cpp import Llama
        from taproot.util import logger

        context_length = self.context_length
        if context_length is None or context_length == 0:
            context_length = self.get_context_length()

        chat_handler = self.get_chat_handler()

        logger.debug(f"Instantiating llama-cpp from {self.model_file} with context length {context_length} and chat format {self.chat_format}")

        self.llama = Llama(
            model_path=self.model_file,
            n_gpu_layers=-1,
            main_gpu=self.device.index,
            n_ctx=context_length,
            verbose=logger.isEnabledFor(logging.DEBUG),
            chat_handler=chat_handler,
            chat_format=self.chat_format,
            tokenizer=self.get_tokenizer()
        )

    def unload(self) -> None:
        """
        Unload the model
        """
        with catch_output() as output:
            del self.llama

    def __call__( # type: ignore[override]
        self,
        *,
        prompt: PromptType,
        image: Optional[ImageType]=None,
        history: Optional[PromptType]=None,
        role: Optional[str]=None,
        seed: SeedType=None,
        max_tokens: Optional[int]=None,
        stop: Optional[Union[str, List[str]]]=None,
        temperature: float=0.2,
        top_p: float=0.95,
        top_k: int=40,
        min_p: float=0.5,
        typical_p: float=1.0,
        presence_penalty: float=0.0,
        frequency_penalty: float=0.0,
        repeat_penalty: float=1.1,
        stream: bool=False,
        use_tools: Union[bool, List[str]]=False,
        return_tool_metadata: bool=False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate text using a large language model.

        :param prompt: The prompt to use.
        :param image: The image to use for models with vision capabilities.
        :param history: Conversation history to prepend to the prompt.
        :param role: The role to use for the conversation.
        :param seed: The seed to use for random generation.
        :param max_tokens: The maximum number of tokens to generate.
        :param stop: The stop tokens to use (i.e., when the model outputs this, consider it complete).
        :param temperature: The temperature to use for generation.
        :param top_p: The nucleus sampling probability.
        :param top_k: The top-k sampling probability.
        :param min_p: The minimum probability to use for sampling.
        :param typical_p: The typical probability to use for sampling.
        :param presence_penalty: The presence penalty to use for sampling.
        :param frequency_penalty: The frequency penalty to use for sampling.
        :param repeat_penalty: The repeat penalty to use for sampling.
        :param stream: Whether to stream the response.
        :param use_tools: Whether to use tools for the task. Allows either true (use all) or a list of tool names to use.
        :param return_tool_metadata: Whether to return tool metadata.
        :return: The generated text.
        """
        tool_choice: Optional[Union[str, Dict[str, Dict[str, str]]]] = None
        tools: Optional[List[LlamaToolMetadata]] = None

        if use_tools:
            tools = self.tool_metadata # List of dicts
            if isinstance(use_tools, list):
                tools = [tool for tool in tools if tool["function"]["name"] in use_tools]
            if self.chat_format is not None and "function" in self.chat_format:
                # LLM can choose the tool
                tool_choice = "auto"
            else:
                 # LLM can't choose the tool, so we use chain-of-thought
                tool_name = self.get_tool(
                    prompt,
                    seed=get_seed(seed),
                    tools=tools
                )
                if tool_name is not None:
                    tool_choice = {"function": {"name": tool_name}}

        remove_image = False
        if image is not None:
            if not self.supports_image:
                raise ValueError("This task does not support images")

            if isinstance(image, str):
                image_path = image
            else:
                images = to_pil_array(image)
                assert len(images) == 1, "Only one image is supported"
                image_path = tempfile.mktemp(suffix=".png")
                images[0].save(image_path)
                remove_image = True
        else:
            image_path = None

        with catch_output() as output:
            try:
                role_object = self.get_role(role)
                conversation = role_object.get_conversation(
                    prompt,
                    image=image_path,
                    history=history
                )
                logger.debug(
                    f"Creating chat generation with conversation: {conversation}, tools {tools}, tool choice {tool_choice}"
                )
                stream_first_generation = False if tool_choice is not None and tool_choice != "auto" else stream
                generation = self.llama.create_chat_completion(
                    messages=self.format_conversation(conversation), # type: ignore[arg-type,unused-ignore]
                    seed=get_seed(seed),
                    max_tokens=max_tokens,
                    stop=self.get_stop_tokens(stop),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    typical_p=typical_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    repeat_penalty=repeat_penalty,
                    stream=stream_first_generation,
                    tool_choice=tool_choice, # type: ignore[arg-type,unused-ignore]
                    tools=tools, # type: ignore[arg-type,unused-ignore]
                )

                logger.debug(f"Chat generation: {generation}")

                if stream_first_generation:
                    for message in generation:
                        completed_text = message["choices"][0]["delta"].get("content", None) # type: ignore[union-attr,index,unused-ignore]
                        if completed_text:
                            self.add_intermediate(completed_text)
                    text_result = self.trim_response(role_object.format_output(self.last_intermediate))
                    if use_tools and return_tool_metadata:
                        return {
                            "function": None,
                            "citations": [],
                            "result": text_result,
                        }
                    return text_result
                else:
                    completed_function_call = generation["choices"][0]["message"].get("function_call", None) # type: ignore[union-attr,index,unused-ignore]
                    if completed_function_call:
                        completed_response = self.execute_and_format_function(
                            prompt,
                            completed_function_call,
                            history=history,
                            seed=get_seed(seed),
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            min_p=min_p,
                            typical_p=typical_p,
                            max_tokens=max_tokens,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            repeat_penalty=repeat_penalty,
                            stream=stream
                        )
                        if stream:
                            for message in completed_response["formatted"]:
                                completed_text = message["choices"][0]["delta"].get("content", None) # type: ignore[attr-defined,index,unused-ignore]
                                if completed_text:
                                    self.add_intermediate(completed_text)
                            text_result = self.trim_response(role_object.format_output(self.last_intermediate))
                            if return_tool_metadata:
                                return {
                                    "function": completed_response["function"],
                                    "citations": completed_response.get("citations", []),
                                    "result": text_result,
                                }
                            return text_result
                        else:
                            completed_text = completed_response["formatted"]["choices"][0]["message"].get("content", None) # type: ignore[attr-defined,index,unused-ignore]
                            text_result = self.trim_response(role_object.format_output(completed_text))
                            if return_tool_metadata:
                                return {
                                    "function": completed_response["function"],
                                    "citations": completed_response.get("citations", []),
                                    "result": text_result,
                                }
                            return text_result
                    else:
                        completed_text = generation["choices"][0]["message"].get("content", None) # type: ignore[attr-defined,index,unused-ignore]
                        if isinstance(completed_text, str):
                            text_result = self.trim_response(role_object.format_output(completed_text))
                        else:
                            text_result = ""
                        if return_tool_metadata:
                            return {
                                "function": None,
                                "citations": [],
                                "result": text_result,
                            }
                        return text_result
            finally:
                out, err = output.output()
                if out:
                    logger.debug(f"Stdout: {out}")
                if err:
                    logger.info(f"stderr (may not be an error): {err}")
                output.clean()
                if image_path is not None and remove_image:
                    import os
                    os.remove(image_path)

class LlamaImageCaptioning(Task):
    """
    A base class for tasks that use the Llama C++ library for image captioning.
    """
    default_describe_prompt = "You are shown an image. Describe the entirety of the image in significant detail, including all objects and subjects in frame and their visual characteristics."
    default_describe_collage = "You are shown an image in two sections: on the left is the entire image with a red square around a section of the image, and on the right is a close-up of the section of the image that was masked. Please describe the contents of the isolated section of the image in as detailed a manner as possible. You should use the context of the larger image to understand what you are seeing in the section, but do not describe parts of the image that are not visible at least partially within the section."
    component_tasks = {
        "llama": LlamaTextGeneration
    }

    @staticmethod
    def trim_caption(caption: str) -> str:
        """
        Trims prefaces from the captioned string to save on tokens.
        """
        trim_section_waterfall = [
            "the",
            "this",
            "isolated",
            "section",
            "of",
            "the",
            "this",
            "image",
            "picture",
            "photo",
            "photograph",
            "features",
            "shows",
            "contains",
            "pictures",
            "depicts",
            "consists",
            "includes",
            "portrays",
            "illustrates",
            "represents",
            "displays",
            "exhibits",
            "of",
            "is",
            "appears",
            "to",
            "show",
            "be",
            "portray",
            "depict",
            "illustrate",
            "represent",
            "display",
            "exhibit",
            "a",
            "close-up",
            "of",
            "a",
            "the",
            "section",
            "that",
            "was",
            "masked",
            "in",
            "of",
            "the",
            "larger",
            "full",
            "complete",
            "left",
            "image",
            "picture",
            "section",
            "showing",
            "containing",
            "encapsulating",
            "in",
            "which",
            "appears",
            "features",
            "by",
            "a",
            "red",
            "square",
        ]
        logger.debug(f"Trimming caption: {caption}")
        for section in trim_section_waterfall:
            section_len = len(section)
            if caption[:section_len].lower() == section:
                caption = caption[section_len+1:]
        caption = caption[0].upper() + caption[1:]
        logger.debug(f"Trimmed caption: {caption}")
        return caption

    def __call__( # type: ignore[override]
        self,
        *,
        image: ImageType,
        prompt: Optional[str]=None,
        tile_size: Optional[int]=None,
        tile_stride: Optional[int]=None,
        max_tokens: int=256,
        description: Optional[str]=None,
        use_collage: bool=False,
    ) -> Union[str, List[Tuple[str, Tuple[int, int, int, int]]]]:
        """
        Gets an image caption, optionally in a tiled manner.

        :param image: The image to caption.
        :param prompt: The prompt to use to override the default captioning prompt.
        :param tile_size: The size of the tiles to use for captioning.
        :param tile_stride: The stride of the tiles to use for captioning.
        :param max_tokens: The maximum number of tokens to generate.
        :param description: A description of the image to help guide the captioning.
        :param use_collage: Whether to use collage-style captioning. This shows the VLM a section of the image and the full image during each tile - for some VLMs, this can improve captioning quality.
        :return: The captioned image or a list of captions and their positions.
        """
        from PIL import Image
        image = to_pil_array(image)[0]
        if tile_size and tile_stride:
            width, height = image.size
            total_tiles = sliding_window_count(
                width=width,
                height=height,
                tile_size=tile_size,
                tile_stride=tile_stride
            )
            captions = []
            for i, tile in maybe_use_tqdm(enumerate(image_tiles(image, tile_size, tile_stride)), total=total_tiles, desc="Captioning tiles"):
                if use_collage:
                    width, height = image.size
                    image_scale = 1 / (height / tile_size)
                    width = int(width * image_scale)
                    scaled_tile_size = int(image_scale * tile_size)
                    # Red image
                    scaled_tile = Image.new("RGB", (scaled_tile_size, scaled_tile_size), (255, 0, 0))
                    # paste tile with 2px border
                    scaled_tile.paste(tile.resize((scaled_tile_size - 4, scaled_tile_size - 4)), (2, 2))
                    x0, y0, x1, y1 = tile.coordinates
                    x0 = int(x0 * image_scale)
                    y0 = int(y0 * image_scale)
                    # overall collage
                    collage = Image.new("RGB", (width + tile_size + 5, tile_size))
                    # Image goes on left
                    collage.paste(image.resize((width, tile_size)))
                    # Scaled-down tile covers section on left
                    collage.paste(scaled_tile, (x0, y0))
                    # Full tile goes on right
                    collage.paste(tile, (width + 5, 0))
                    prompt = self.default_describe_collage
                    if prompt is None:
                        prompt = self.default_describe_collage
                elif prompt is None:
                    prompt = self.default_describe_prompt

                if description is not None:
                    describe = f"{prompt}. A human has provided the following description of the entire image, use this to inform your answer: '{description}'"
                else:
                    describe = prompt

                captions.append({
                    "prompt": self.trim_caption(
                        self.tasks.llama(
                            prompt=describe,
                            image=tile,
                            max_tokens=max_tokens
                        )
                    ),
                    "position": list(tile.coordinates)
                })
                logger.debug(f"Captioned tile {i}: {captions[-1]}")
            return captions # type: ignore[return-value]
        else:
            prompt = self.default_describe_prompt if prompt is None else prompt
            if description is not None:
                describe = f"{prompt}. A human has provided the following description, use this to inform your answer: '{description}'"
            else:
                describe = prompt

            return self.trim_caption(
                self.tasks.llama(
                    prompt=describe,
                    image=image,
                    max_tokens=max_tokens
                )
            )
