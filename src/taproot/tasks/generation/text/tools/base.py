from __future__ import annotations

import requests
import warnings

from typing import List, Dict, Type, Any, Optional
from typing_extensions import TypedDict, NotRequired

from taproot.util import (
    IntrospectableMixin,
    multiline_trim,
    get_parameter_enum,
    package_is_available,
    get_secret,
    logger,
)

__all__ = [
    "Tool",
    "LlamaParameterMetadata",
    "LlamaParametersMetadata",
    "LlamaFunctionMetadata",
    "LlamaToolMetadata",
    "LlamaToolCall",
    "LlamaToolFunctionCall",
    "LlamaToolFunctionCallResult",
    "LlamaToolFunctionCallResultCitation",
]

# LLaMA metadata types

class LlamaToolFunctionCall(TypedDict):
    """
    Function call format required by llama .
    """
    name: str
    arguments: str # JSON-encoded arguments

class LlamaToolCall(TypedDict):
    """
    Call format required by llama.
    """
    id: str
    type: str # 'function'
    function: LlamaToolFunctionCall

class LlamaParameterMetadata(TypedDict):
    """
    Metadata format required by llama .
    """
    type: str # 'string' | 'number' | 'boolean' | 'object' | 'array'
    description: NotRequired[str]
    enum: NotRequired[List[str]]

class LlamaParametersMetadata(TypedDict):
    """
    Metadata format required by llama .
    """
    type: str # 'object'
    properties: Dict[str, LlamaParameterMetadata]
    required: List[str]

class LlamaFunctionMetadata(TypedDict):
    """
    Metadata format required by llama .
    """
    name: str
    description: str
    parameters: LlamaParametersMetadata

class LlamaToolMetadata(TypedDict):
    """
    Metadata format required by llama .
    """
    type: str # "function"
    function: LlamaFunctionMetadata

# Taproot metadata types

class LlamaToolFunctionCallResultCitation(TypedDict):
    """
    A citation from a tool result.
    """
    url: str
    title: NotRequired[Optional[str]]
    source: NotRequired[Optional[str]]

class LlamaToolFunctionCallResult(TypedDict):
    """
    A result from a tool call.
    """
    result: Any
    formatted: NotRequired[str]
    function: Optional[LlamaToolFunctionCall] # None if no tool selected
    citations: NotRequired[List[LlamaToolFunctionCallResultCitation]]

class Tool(IntrospectableMixin):
    """
    A base class for all tools.
    Tools are methods that are able to be executed by supported function-calling LLMs.

    Tools should be stateless and should not have any side-effects.
    MOST tools should be read-only. If you are creating a tool that can do stateful operations,
    you should be sure to mitigate any damage potentially caused by the LLM generating erroneous
    input, or the user attempting to convince the LLM to do something it shouldn't.

    Great ideas for tools all augment the user's experience in some way, or provide a useful
    utility that is not reliably provided by the LLM itself.
    """
    tool_name: str
    api_key_var: Optional[str] = None
    proxy_var: Optional[str] = "TAPROOT_PROXY"
    required_packages: List[str] = []
    _llama_metadata: LlamaToolMetadata
    _citations: List[LlamaToolFunctionCallResultCitation]

    @classmethod
    def llama_metadata(cls) -> LlamaToolMetadata:
        """
        Returns a description of the tool.
        """
        if not hasattr(cls, "_llama_metadata"):
            tool_name = getattr(cls, "tool_name", None)
            if tool_name is None:
                raise ValueError("The tool_name attribute must be set on the tool.")
            metadata = cls.introspect()
            description = metadata.get("short_description", metadata.get("long_description", ""))
            if not description:
                warnings.warn(f"No description found for tool {tool_name} in docstring. This can cause the LLM to not understand the tool; please add a description.")

            parameters: LlamaParametersMetadata = {
                "type": "object",
                "properties": {},
                "required": []
            }

            for parameter_name, parameter in metadata.get("parameters", {}).items():
                parameter_type = parameter["parameter_type"]
                parameter_description = parameter.get("description", None)
                parameter_enum = get_parameter_enum(parameter_type)

                if parameter_type is None:
                    raise ValueError(f"Parameter type must be set for parameter {parameter_name} in tool {tool_name}.")
                elif not isinstance(parameter_type, type):
                    raise RuntimeError(f"Introspection failed for parameter {parameter_name} (type \"{parameter_type}\" {type(parameter_type)}) in tool {tool_name}.")
                elif parameter_type is str:
                    parameter_type_name = "string"
                elif parameter_type is int or parameter_type is float:
                    parameter_type_name = "number"
                elif parameter_type is bool:
                    parameter_type_name = "boolean"
                elif parameter_type is dict or parameter_type is list:
                    raise ValueError(f"Parameter type must be a primitive type for parameter {parameter_name} in tool {tool_name}.")
                else:
                    raise NotImplementedError(f"Unhandled parameter type {parameter_type} for parameter {parameter_name} in tool {tool_name}.")

                parameters["properties"][parameter_name] = {
                    "type": parameter_type_name,
                }

                if parameter_description is not None:
                    parameters["properties"][parameter_name]["description"] = parameter_description

                if parameter_enum is not None:
                    parameters["properties"][parameter_name]["enum"] = parameter_enum

                if parameter["required"]:
                    parameters["required"].append(parameter_name)

            cls._llama_metadata = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": parameters
                }
            }

        return cls._llama_metadata

    @classmethod
    def enumerate(cls) -> Dict[str, Type[Tool]]:
        """
        Enumerates all tools in the module.
        """
        tools: Dict[str, Type[Tool]] = {}
        for tool in cls.__subclasses__():
            if getattr(tool, "tool_name", None) is None:
                logger.debug(f"Tool {tool.__name__} does not have a tool_name attribute. Skipping.")
            elif not tool.is_available():
                logger.debug(f"Tool {tool.tool_name} is not available. Skipping.")
            else:
                if tool.tool_name in tools:
                    logger.error(f"Duplicate tool name {tool.tool_name} found in module {tool.__module__}. It will overwrite the previous tool.")
                tools[tool.tool_name] = tool
            for sub_tool_name, sub_tool in tool.enumerate().items():
                if sub_tool_name in tools:
                    logger.error(f"Duplicate tool name {sub_tool_name} found in module {sub_tool.__module__}. It will overwrite the previous tool.")
                tools[sub_tool_name] = sub_tool
        return tools

    @classmethod
    def is_available(cls) -> bool:
        """
        Returns whether the tool is available.
        """
        if not all(package_is_available(package) for package in cls.required_packages):
            return False
        if cls.api_key_var is not None and cls.get_api_key() is None:
            return False
        return True

    @classmethod
    def get_api_key(cls) -> Optional[str]:
        """
        Returns the API key for the tool.
        """
        if cls.api_key_var is None:
            return None
        return get_secret(cls.api_key_var)

    @classmethod
    def get_proxy(cls) -> Optional[str]:
        """
        Returns the proxy for the tool.
        """
        if cls.proxy_var is None:
            return None
        return get_secret(cls.proxy_var)

    @property
    def api_key(self) -> Optional[str]:
        """
        Returns the API key for the tool.
        """
        if getattr(self, "_api_key", None) is None:
            self._api_key = self.get_api_key()
        return self._api_key

    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        """
        Sets the API key for the tool.
        """
        self._api_key = value

    @property
    def proxy(self) -> Optional[str]:
        """
        Returns the proxy for the tool.
        """
        if getattr(self, "_proxy", None) is None:
            self._proxy = self.get_proxy()
        return self._proxy

    @proxy.setter
    def proxy(self, value: Optional[str]) -> None:
        """
        Sets the proxy for the tool.
        """
        self._proxy = value
        if hasattr(self, "_session"):
            if value is None:
                self._session.proxies.pop("http", None)
                self._session.proxies.pop("https", None)
            else:
                self._session.proxies["http"] = self._session.proxies["https"] = value

    @property
    def citations(self) -> List[LlamaToolFunctionCallResultCitation]:
        """
        Returns the citations for the tool.
        """
        return getattr(self, "_citations", [])

    @property
    def session(self) -> requests.Session:
        """
        Gets a requests session.
        """
        if not hasattr(self, "_session"):
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"})
            if self.proxy is not None:
                self._session.proxies["http"] = self._session.proxies["https"] = self.proxy
        return self._session

    def cite(
        self,
        url: str,
        title: Optional[str]=None,
        source: Optional[str]=None
    ) -> None:
        """
        Cites a source.
        """
        if not hasattr(self, "_citations"):
            self._citations = []
        self._citations.append({
            "url": url,
            "title": title.strip() if title is not None else None,
            "source": source.strip() if source is not None else None
        })

    def read(self, url: str, cite: bool=False) -> str:
        """
        Reads a URL and extracts text.
        """
        from bs4 import BeautifulSoup
        response = self.session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        contents = multiline_trim(soup.get_text())
        if cite:
            self.cite(
                url,
                title = soup.title.string.strip() if soup.title is not None and soup.title.string is not None else None
            )
        return contents

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        The method that is called when the tool is executed.
        """
        raise NotImplementedError("This tool has not been implemented yet.")
