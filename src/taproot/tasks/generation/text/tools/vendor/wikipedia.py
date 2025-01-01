from typing import Optional, List, Iterator, Tuple, Any
from datetime import datetime

from ..base import Tool

__all__ = ["WikipediaTool"]

class WikipediaTool(Tool):
    """
    A tool for looking up information on Wikipedia.
    """
    tool_name = "wikipedia"
    ignored_sections = {"references", "external_links", "see_also", "notes"}
    ignored_keywords = {"", "wikipedia"}
    required_packages = ["bs4"]

    def search(
        self,
        keywords: List[Optional[str]],
        max_results_per_keyword: int=1,
    ) -> Iterator[Tuple[str, str, str]]:
        """
        Search Wikipedia for a given set of keywords.
        """
        keywords = [kw for kw in keywords if kw is not None and kw.lower() not in self.ignored_keywords]
        for i in range(len(keywords)):
            joined_keywords = ", ".join(["" if k is None else k for k in keywords[:i + 1]])
            search_response = self.session.get(
                "https://api.wikimedia.org/core/v1/wikipedia/en/search/page",
                params={"q": joined_keywords, "limit": max_results_per_keyword}, # type: ignore[arg-type]
            )
            search_response.raise_for_status()
            search_pages = search_response.json().get("pages", [])
            for page in search_pages:
                yield joined_keywords, page["title"], page["key"]

    def get_page_content(self, page_key: str) -> str:
        """
        Get the content of a Wikipedia page.
        """
        from bs4 import BeautifulSoup
        page_content = self.session.get(
            f"https://api.wikimedia.org/core/v1/wikipedia/en/page/{page_key}/html",
        )
        page_content.raise_for_status()
        soup = BeautifulSoup(page_content.text, "html.parser")
        for section in soup.find_all("section"):
            section_header = section.find("h2")
            if section_header is not None:
                section_id = section_header.get("id")
                if section_id.lower() in self.ignored_sections:
                    section.decompose()
                    continue
            for child in section.children:
                if child.name not in {"p", "ul", "ol", "h2", "h3"}:
                    child.replace_with("")
            for superscript_ref in section.find_all("sup", class_="reference"):
                superscript_ref.replace_with("")
        return soup.get_text()

    def __call__(
        self,
        keyword: str,
        keyword_2: Optional[str]=None,
        keyword_3: Optional[str]=None,
        keyword_4: Optional[str]=None,
        keyword_5: Optional[str]=None,
        **kwargs: Any,
    ) -> str:
        """
        Look up information on Wikipedia using a search query. Wikipedia is a free online encyclopedia containing information on a wide variety of topics.

        :param keyword: The primary key words to look up. For best results, ask for a specific event, individual, or topic.
        :param keyword_2: A secondary key word to help narrow the search. Optional.
        :param keyword_3: A tertiary key word to help narrow the search. Optional.
        :param keyword_4: A quaternary key word to help narrow the search. Optional.
        :param keyword_5: A quinary key word to help narrow the search. Optional.
        :return: The page content as a string.
        """
        if keyword is None:
            return "No search query provided."

        accessed_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        keywords = [keyword, keyword_2, keyword_3, keyword_4, keyword_5]
        page_summary = None

        for kw, title, key in self.search(keywords):
            page_url = f"https://en.wikipedia.org/wiki/{key}"
            page_label = f"Results for [{kw}]: “{title}” accessed on {accessed_on} at {page_url}"
            self.cite(page_url, title)
            page_content = self.get_page_content(key)
            if page_summary is None:
                page_summary = f"{page_label}\n{page_content}"
            else:
                page_summary += f"\n\n{page_label}\n{page_content}"

        assert page_summary is not None, "No results found."
        return page_summary
