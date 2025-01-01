from __future__ import annotations

import re

from typing import List, Optional, Iterator, Type, Any, TYPE_CHECKING

__all__ = ["Role"]

if TYPE_CHECKING:
    from taproot.hinting import MessageDict, PromptType

class Role:
    """
    This class allows for defining roles for LLMs to adopt
    """
    role_name = "default"
    remove_parentheticals = True
    single_line = False
    use_system = False

    @classmethod
    def enumerate(cls) -> Iterator[Type[Role]]:
        """
        Enumerates all roles.
        """
        for role_class in cls.__subclasses__():
            role_class_name = getattr(role_class, "role_name", None)
            if role_class_name is not None:
                yield role_class

    @classmethod
    def get(cls, name: str) -> Type[Role]:
        """
        Gets a role by name
        """
        tried_classes: List[str] = []
        for role_class in cls.enumerate():
            if role_class.role_name == name:
                return role_class
            tried_classes.append(role_class.role_name)
        tried_classes_string = ", ".join(tried_classes)
        raise ValueError(f"Could not find role by name {name} (found {tried_classes_string})")

    def get_conversation(
        self,
        prompt: PromptType,
        image: Optional[str]=None,
        history: Optional[PromptType]=None,
        **kwargs: Any
    ) -> List[MessageDict]:
        """
        Gets the conversation for the role, standardizing the prompts
        """
        if not isinstance(prompt, list):
            prompt_list = [prompt]
        else:
            prompt_list = prompt # type: ignore[assignment]
        if history is not None:
            if not isinstance(history, list):
                history = [history] # type: ignore[assignment]
            if len(history) % 2 != 0:
                raise ValueError("History must have an even number of elements")
            prompt_list = history + prompt_list # type: ignore[assignment,operator]

        num_prompts = len(prompt_list)
        conversation: List[MessageDict] = []

        if self.use_system:
            conversation = self.system_conversation

        for i, prompt_item in enumerate(prompt_list):
            if isinstance(prompt_item, str):
                prompt_dict: MessageDict = {
                    "text": prompt_item,
                    "image": None if i != num_prompts - 1 else image
                }
            else:
                prompt_dict = prompt_item

            if "role" not in prompt_dict:
                if i == 0:
                    prompt_dict["role"] = "user"
                else:
                    prompt_dict["role"] = "user" if conversation[-1]["role"] == "assistant" else "assistant"
            if i >= num_prompts - 2:
                if prompt_dict["role"] == "user":
                    prompt_dict["text"] = self.format_input(prompt_dict["text"], **kwargs)
                elif prompt_dict["role"] == "assistant":
                    prompt_dict["text"] = self.format_output(prompt_dict["text"], **kwargs)
            conversation.append(prompt_dict)
        return conversation

    def format_input(self, message: Optional[str], **kwargs: Any) -> str:
        """
        Given user input, format the message to the bot
        """
        return "" if message is None else message

    def format_output(self, message: str, **kwargs: Any) -> str:
        """
        Given bot output, format the message to the user
        """
        if self.remove_parentheticals:
            # Remove beginning and ending parentheticals. Some LLM's like to add
            # a caveat to the beginning like (in a happy tone) and notes to the
            # end like (note: this is a joke). This removes those for a more natural
            # conversational cadence.
            message = message.strip()
            start_pattern = re.compile(r'^\([^()]*\)\s*')
            end_pattern = re.compile(r'\s*\([^()]*\)$')

            # Remove leading parentheticals repeatedly
            while True:
                new_message = start_pattern.sub('', message)
                if new_message == message:
                    break
                message = new_message.strip()

            # Remove trailing parentheticals repeatedly
            while True:
                new_message = end_pattern.sub('', message)
                if new_message == message:
                    break
                message = new_message.strip()

        if self.single_line and message:
            message = message.splitlines()[0]

        return message

    """Mutables"""

    @property
    def introduction(self) -> str:
        """
        Gets either the set or default introduction
        """
        if hasattr(self, "_introduction"):
            return self._introduction
        return self.system_introduction

    @introduction.setter
    def introduction(self, introduction: Optional[str]) -> None:
        """
        Sets the introduction (or resets to default)
        """
        if not introduction:
            try:
                delattr(self, "_introduction")
            except AttributeError:
                pass
        else:
            self._introduction = introduction

    """Immutables"""

    @property
    def system_greeting(self) -> str:
        """
        The greeting displayed at the start of conversations
        """
        return "Hello, I am your personal assistant. How can I help today?"

    @property
    def system_introduction(self) -> str:
        """
        The message told to the bot at the beginning instructing it
        """
        return "You are a personal assistant. You are here to help the user with their needs."

    @property
    def system_rules(self) -> List[str]:
        """
        The rules given to the regular bot.
        """
        return []

    @property
    def system_examples(self) -> List[MessageDict]:
        """
        Examples given to the system to refine behavior
        """
        return []

    @property
    def system_message(self) -> str:
        """
        The formatted message told to the bot at the beginning instructing it
        """
        message = self.introduction
        if self.system_rules:
            message += "\n\nThere are a few rules to follow:\n{0}".format(
                "\n".join([f"  - {rule}" for rule in self.system_rules])
            )
        return message

    @property
    def system_conversation(self) -> List[MessageDict]:
        """
        The conversation to begin the invocation
        """
        return [{
            "role": "system",
            "text": self.system_message,
            "image": None
        }] + self.system_examples # type: ignore[return-value]
