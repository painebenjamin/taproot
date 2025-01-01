from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duckduckgo_search import DDGS

from taproot.util import logger, get_top_level_domain_from_url
from ..base import Tool

__all__ = [
    "DuckDuckGoSearchTool",
    "DuckDuckGoSearchReadTool",
    "DuckDuckGoNewsTool",
    "DuckDuckGoNewsReadTool",
    "DuckDuckGoHeadlinesTool",
]

class DuckDuckGoTool(Tool):
    """
    A parent class for DuckDuckGo tools.
    """
    @property
    def ddgs(self) -> DDGS:
        """
        Returns an instance of the DuckDuckGo search tool.

        :return: The DuckDuckGo search tool.
        """
        if not hasattr(self, "_ddgs"):
            from duckduckgo_search import DDGS
            self._ddgs = DDGS(proxy=self.proxy)
        return self._ddgs

class DuckDuckGoSearchTool(DuckDuckGoTool):
    """
    A class for searching DuckDuckGo.
    """
    tool_name = "web-search"
    required_packages = ["duckduckgo_search"]

    def __call__(
        self,
        search: str,
    ) -> str:
        """
        Searches the internet using DuckDuckGo for the given search term and returns a brief summary for the first ten results. For best results, use specific search terms.

        :param search: The search term.
        :return: The search results.
        """
        results = self.ddgs.text(search, max_results=10)
        for result in results:
            self.cite(
                result["href"],
                title=result["title"],
                source=result.get("source", None)
            )
        return str(results)

class DuckDuckGoSearchReadTool(DuckDuckGoTool):
    """
    A class for searching DuckDuckGo and reading the first result.
    """
    tool_name = "web-search-read"
    ignored_domains = {"duckduckgo.com", "msn.com"}
    required_packages = ["duckduckgo_search"]

    def __call__(
        self,
        search: str,
    ) -> str:
        """
        Searches the internet using DuckDuckGo for the given search term and reads the first result. For best results, use specific search terms.

        :param search: The search term.
        :return: The search results.
        """
        results = self.ddgs.text(search, max_results=10)
        i = 0
        while get_top_level_domain_from_url(results[i]["href"]) in self.ignored_domains:
            i += 1
        if i >= len(results):
            return "No useful results found."
        self.cite(
            results[i]["href"],
            title=results[i]["title"],
            source=results[i].get("source", None)
        )
        return self.read(results[i]["href"])

class DuckDuckGoNewsTool(DuckDuckGoTool):
    """
    A class for searching DuckDuckGo news.
    """
    tool_name = "news-search"
    required_packages = ["duckduckgo_search"]

    def __call__(
        self,
        search: str,
    ) -> str:
        """
        Searches DuckDuckGo news for the given search term and returns a brief summary for the first ten results. Use this tool to provide up-to-date news on a specific topic. For best results, use specific search terms.

        :param search: The search term.
        :return: The search results.
        """
        results = self.ddgs.news(search, max_results=10)
        for result in results:
            self.cite(
                result["url"],
                title=result["title"],
                source=result.get("source", None)
            )
        return str(results)

class DuckDuckGoNewsReadTool(DuckDuckGoTool):
    """
    A class for searching DuckDuckGo news and reading the first result.
    """
    tool_name = "news-read"
    ignored_domains = {"duckduckgo.com", "msn.com"}
    required_packages = ["duckduckgo_search", "bs4"]

    def __call__(
        self,
        search: str,
    ) -> str:
        """
        Searches DuckDuckGo news for the given search term and reads the first result. Use this tool to provide up-to-date news on a specific topic.

        :param search: The search term.
        :return: The search results.
        """

        results = self.ddgs.news(search, max_results=10)
        i = 0
        while get_top_level_domain_from_url(results[i]["url"]) in self.ignored_domains:
            i += 1
        if i >= len(results):
            return "No useful results found."
        logger.info(f"Getting news from {results[i]['url']}")
        self.cite(
            results[i]["url"],
            title=results[i]["title"],
            source=results[i].get("source", None)
        )
        return self.read(results[i]["url"])

class DuckDuckGoHeadlinesTool(DuckDuckGoTool):
    """
    Gets the latest headlines from DuckDuckGo.
    """
    tool_name = "news-headlines"
    required_packages = ["duckduckgo_search"]

    def __call__(
        self,
        period: str = "day",
        num_headlines: int = 10,
    ) -> str:
        """
        Gets the latest news headlines from DuckDuckGo without a specific search term. Use this tool when asked for general news or current events.

        :param period: The period of time to get the headlines from. Use either 'day', 'week', 'month', or 'year'. Defaults to 'day'.
        :param num_headlines: The number of headlines to get. Defaults to 10.
        :return: The headlines.
        """
        from duckduckgo_search import DDGS
        time = "d" if period == "day" else "w" if period == "week" else "m" if period == "month" else None
        results = self.ddgs.news("news", max_results=num_headlines, timelimit=time)
        for result in results:
            self.cite(
                result["url"],
                title=result["title"],
                source=result["source"]
            )
        return "\n".join([
            f"{result['title']} - {result['date']}"
            for result in results
        ])
