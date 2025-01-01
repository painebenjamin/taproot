from __future__ import annotations

from typing import Any, List, Optional, TYPE_CHECKING

from .base import Role

if TYPE_CHECKING:
    from taproot.hinting import MessageDict

__all__ = ["ToolPicker", "ToolResultFormatter"]

class ToolPicker(Role):
    """
    A role that is responsible for selecting the appropriate tool for a given task.
    """
    role_name = "tool-picker"
    use_system = True

    @property
    def base_role(self) -> Role:
        """
        A singular role instance for helper methods and properties.
        """
        if getattr(self, "_base_role", None) is None:
            self._base_role = Role()
        return self._base_role

    @property
    def system_introduction(self) -> str:
        """
        The introduction given to the system to introduce the role.
        """
        return "You are part of a team of bots assisting users with varying requests. " \
               "Your role is to determine if a user is requesting a tool and, if so, " \
               "select the appropriate tool for the user."

    def format_input(self, message: Optional[str], **kwargs: Any) -> str:
        """
        Format the input for the system.
        """
        available_tools = kwargs.get("available_tools", None)
        history = kwargs.get("conversation_history", None)

        if isinstance(available_tools, list):
            # Just names
            available_tool_text = "\n".join(available_tools)
        elif isinstance(available_tools, dict):
            # Name and description
            available_tool_text = "\n".join([f"{name}: {description}" for name, description in available_tools.items()])
        else:
            available_tool_text = "None"

        if history is None:
            conversation_text = f"[user]: {message}"
        else:
            conversation_text = "\n".join([
                f"[{message_dict['role']}]: {message_dict['text']}"
                for message_dict in self.base_role.get_conversation("" if message is None else message, history=history)
            ])

        return f"""= Toolbox =====================
{available_tool_text}
= Conversation =================
{conversation_text}
================================
Based on the toolbox and conversation above, select the appropriate tool for the latest user input, or 'None' if no tool is needed."""

    @property
    def system_rules(self) -> List[str]:
        """
        The rules given to the system to help it select the appropriate tool.
        """
        return [
            "The available tools will be provided to you with each request.",
            "You should select the tool that best matches the user's request.",
            "When the user is not requesting a tool, you should not select a tool, and instead respond with 'None'.",
            "ONLY select a tool if the user specifically requests it or it is NECESSARY to fulfill the user's request. When presented with a vague request that does not pertain to any particular tool, respond with 'None'.",
            "If the user is requesting a tool in the future, you should not select a tool until the user makes a subsequent request.",
            "Respond only with the name of a tool or 'None', do not clarify or provide additional information."
        ]

    @property
    def system_examples(self) -> List[MessageDict]:
        """
        Example interactions between the user and the system.
        """
        simple_tools = {
            "weather": "Get the current and forecasted weather for a location.",
            "wolfram-alpha": "Perform simple or complex calculations using plain language.",
            "news": "Get the latest news headlines and articles.",
            "horoscope": "Get a daily or forecasted horoscope for a zodiac sign.",
            "date-time": "Get the current date and time."
        }
        generative_tools = {
            "image": "Generate an image based on a text description.",
            "video": "Generate a video based on a text description.",
            "speech": "Generate speech based on a text description.",
            "sound": "Generate sound based on a text description, like ambient noise or sound effects.",
            "music": "Generate music based on a text description, like a genre or mood, as well as lyrics or themes."
        }
        lookup_tools = {
            "dictionary": "Look up the definition, pronunciation, and usage of a word.",
            "thesaurus": "Look up synonyms and antonyms for a word.",
            "wikipedia": "Look up information on a topic from Wikipedia.",
            "wikigames": "Look up information on video games from multiple fan-made wikis, including Fandom and Fextralife.",
            "mayo-clinic": "Look up information on medical conditions, symptoms, and treatments from Mayo Clinic."
        }
        news_tools = {
            "news-headlines": "Look up the latest news headlines.",
            "news-search": "Search for news articles on a specific topic.",
            "news-read": "Search for a news article and read it in full."
        }

        return [
            {
                "role": "user",
                "text": self.format_input(
                    "Hello, how are you?",
                    available_tools=simple_tools
                )
            },
            {
                "role": "assistant",
                "text": "None"
            },
            {
                "role": "user",
                "text": self.format_input(
                    "What's the weather like in Fresno today?",
                    available_tools=simple_tools,
                    conversation_history=["Hello, how are you?", "Hello! I'm doing well, thank you for asking. How can I help you today?"]
                )
            },
            {
                "role": "assistant",
                "text": "weather"
            },
            {
                "role": "user",
                "text": self.format_input(
                    "Can you show me a picture of a cat?",
                    available_tools=generative_tools
                )
            },
            {
                "role": "assistant",
                "text": "image"
            },
            {
                "role": "user",
                "text": self.format_input(
                    "jazz music",
                    available_tools=lookup_tools,
                )
            },
            {
                "role": "assistant",
                "text": "None",
            },
            {
                "role": "user",
                "text": self.format_input(
                    "portland",
                    available_tools=lookup_tools,
                )
            },
            {
                "role": "assistant",
                "text": "None",
            },
            {
                "role": "user",
                "text": self.format_input(
                    "Do you have the current time?",
                    available_tools=simple_tools,
                    conversation_history=["Hello, how are you?", "Hello! I'm doing well, thank you for asking. How can I help you today?"]
                )
            },
            {
                "role": "assistant",
                "text": "date-time"
            },
            {
                "role": "user",
                "text": "Thank you!"
            },
            {
                "role": "assistant",
                "text": "None"
            },
            {
                "role": "user",
                "text": self.format_input(
                    "What's the latest news?",
                    available_tools=news_tools,
                )
            },
            {
                "role": "assistant",
                "text": "news-headlines"
            },
            {
                "role": "user",
                "text": self.format_input(
                    "Is there any news about the upcoming election?",
                    available_tools=news_tools,
                )
            },
            {
                "role": "assistant",
                "text": "news-search"
            },
            {
                "role": "user",
                "text": self.format_input(
                    "What does the word 'ubiquitous' mean?",
                    available_tools=lookup_tools,
                    conversation_history=[
                        "Please provide five synonyms for the word 'quick'.",
                        "Sure! Here are five synonyms for the word 'quick': fast, speedy, rapid, swift, and prompt."
                    ]
                )
            },
            {
                "role": "assistant",
                "text": "dictionary"
            },
            {
                "role": "user",
                "text": self.format_input(
                    "What is the meaning of life?",
                    available_tools=lookup_tools
                )
            },
            {
                "role": "assistant",
                "text": "None"
            },
            {
                "role": "user",
                "text": self.format_input(
                    "For the remainder of the conversation, I'd like you to generate everything I type as speech using the 'Sarah' voice.",
                    available_tools=generative_tools
                )
            },
            {
                "role": "assistant",
                "text": "None"
            },
            {
                "role": "user",
                "text": self.format_input(
                    "Tell me a joke",
                    available_tools=generative_tools,
                    conversation_history=[
                        "For the remainder of the conversation, I'd like you to generate everything I type as speech using the 'Sarah' voice.",
                        "Okay! I'll generate everything you type as speech using the 'Sarah' voice."
                    ]
                )
            },
            {
                "role": "assistant",
                "text": "speech"
            },
            {
                "role": "user",
                "text": self.format_input(
                    "Give me a play-by-play of the fight from last night.",
                    available_tools=news_tools,
                    conversation_history=[
                        "Tell me a joke",
                        "Why did the scarecrow win an award? Because he was outstanding in his field!"
                    ]
                )
            },
            {
                "role": "assistant",
                "text": "news-read"
            }
        ]

class ToolResultFormatter(Role):
    """
    The role that is responsible for formatting the results of a tool
    to make it flow more naturally in a conversation.
    """
    role_name = "tool-result-formatter"
    use_system = True

    @property
    def system_introduction(self) -> str:
        """
        The introduction given to the system to introduce the role.
        """
        return "You are part of a team of bots assisting users with varying requests. " \
               "Your role is to take the result of a tool and use it to answer a specific user inquiry. " \
               "This can involve identifying one or more pieces of information from the result, or simply " \
               "rephrasing the result in a more conversational manner, depending on the user's specific request."

    def format_input(
        self,
        message: Optional[str],
        tool: Optional[str]=None,
        result: Optional[str]=None,
        **kwargs: Any
    ) -> str:
        """
        Format the input for the system.
        """
        return f"For the user input '{message}' and tool '{tool}', format the result '{result}'."

    @property
    def system_rules(self) -> List[str]:
        """
        The rules given to the system to help it format the results of a tool.
        """
        return [
            "If the user asked a specific question that the result of the tool answers, you should focus first on directly answering the user's inquiry.",
            "You should format your response in a way that makes it flow naturally in the conversation.",
            "The user will not see the original result - do NOT reference the existence of the original result or the tool in your response. For example, do NOT say 'You asked me to summarize...' or 'The tool said...'.",
            "You are allowed to add commentary or additional information after repeating the substantive information.",
            "It is possible that the information presented does not fulfill the user's request, due to the nature of the task. In these cases, you should still format the information in a conversational manner, but you may call attention to the fact that the information may not be directly relevant to the user's request.",
            "Use the other messages in the conversation for an example of the style and tone to use.",
        ]

    @property
    def system_examples(self) -> List[MessageDict]:
        """
        Example interactions between the user and the system.
        """
        return [
            {
                "role": "user",
                "text": "Hello, how are you?"
            },
            {
                "role": "assistant",
                "text": "Hello! I'm doing well, thank you for asking. How can I help you today?"
            },
            {
                "role": "user",
                "text": self.format_input("What does the word 'ubiquitous' mean?", tool="dictionary", result="Being or seeming to be everywhere at the same time")
            },
            {
                "role": "assistant",
                "text": "The word 'ubiquitous' means being or seeming to be everywhere at the same time. Like how the internet is everywhere these days!"
            },
            {
                "role": "user",
                "text": self.format_input("Could you please look up the menu for Turbo Hot Dogs and let me know if they have vegan options?", tool="web-search", result="Welcome to the website for Turbo Hot Dogs!\nAbout Us - Menu - Locations - Contact Us\nTurbo Dogs is more than just a hot dog stand. We are an experience to behold. We've been named the best Hot Dogs in New York for the past 32 years, ever since our founder Robert 'Turbo' Taylor opened our first stand in 1989. Stop by today and experience the Turbo dog!\nLocations\n1127 Grover Cleveland Hwy., Turboland, NY, USA\nPhone: (212)-664-7665\nEmail: contact@turbohotdogs.com\nHours: Mon-Sun 10am-10pm\nCopyright © 2021 Turbo Hot Dogs")
            },
            {
                "role": "assistant",
                "text": "I found the website for Turbo Hot Dogs, but I couldn't find a menu or information about vegan options. It seems like they're more focused on the experience of eating a Turbo dog! You might want to give them a call to ask about vegan options - their phone number is (212)-664-7665."
            },
            {
                "role": "user",
                "text": "Thank you!"
            },
            {
                "role": "assistant",
                "text": "You're welcome! Let me know if you need anything else."
            },
            {
                "role": "user",
                "text": self.format_input("What's the weather like in Fresno today?", tool="weather", result="Sunny, 60 to 95 degrees, 21% humidity, 5 mph winds")
            },
            {
                "role": "assistant",
                "text": "The weather in Fresno today is sunny with a low of 60 degrees and a high of 95 degrees. The humidity is at 21% and the winds are blowing at 5 mph. It's a beautiful day! Be sure to wear sunscreen if you're going outside, and stay hydrated! It's warm out there!"
            }
        ]
