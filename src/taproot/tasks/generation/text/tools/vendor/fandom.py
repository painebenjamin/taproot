import re
import urllib

from typing import Optional, Iterator, Tuple
from datetime import datetime
from taproot.util import multiline_trim, logger

from ..base import Tool

__all__ = ["FandomTool"]

class FandomTool(Tool):
    """
    A tool for looking up information on Fandom, an expansive and detailed wiki site for all things gaming, movies, TV, and anime.
    """
    tool_name = "fandom"
    ignored_sections = {"see also", "references", "categories", "languages", "external links", "navigation"}
    required_packages = ["bs4"]

    def find_community(self, search: str) -> Optional[str]:
        """
        Find a Fandom community by name.
        """
        from bs4 import BeautifulSoup
        search_response = self.session.get(
            "https://community.fandom.com/wiki/Special:Search",
            params={"query": urllib.parse.quote_plus(search), "scope": "cross-wiki"},
        )
        search_response.raise_for_status()
        search_soup = BeautifulSoup(search_response.text, "html.parser")
        community_result = search_soup.find("div", class_="unified-search__community")
        if community_result is not None:
            return community_result.find("a").get("href") # type: ignore[union-attr,return-value]
        return None

    def search(
        self,
        search: str,
        community: Optional[str]=None,
        max_results_per_search: int=1,
    ) -> Iterator[Tuple[str, str]]:
        """
        Search for a query on Fandom.

        :param search: The search query.
        :param community: The Fandom community to search in. Optional.
        :param max_results_per_search: The maximum number of results to return per search. Default is 1.
        :return: An iterator of tuples containing the search result title and URL.
        """
        from bs4 import BeautifulSoup
        params = {"query": urllib.parse.quote_plus(search)}
        if community is None:
            search_url = "https://community.fandom.com/wiki/Special:Search"
            params["scope"] = "cross-wiki"
        else:
            search_url = f"{community}/wiki/Special:Search"
            scope = None
        search_response = self.session.get(
            search_url,
            params=params,
            allow_redirects=False,
        )
        search_response.raise_for_status()
        if search_response.status_code == 301:
            # Search may have moved, follow it
            search_response = self.session.get(
                search_response.headers["Location"],
                allow_redirects=False,
            )
            search_response.raise_for_status()

        if search_response.status_code == 302:
            # The search responded with a single result, this usually
            # happens when the query matches a result exactly
            yield search, search_response.headers["Location"]
        else:
            search_soup = BeautifulSoup(search_response.text, "html.parser")
            search_results = search_soup.find_all("li", class_="unified-search__result")
            for result in search_results[:max_results_per_search]:
                result_title = result.find("a", class_="unified-search__result__title")
                result_url = result_title.get("href")
                result_title = multiline_trim(result_title.get_text())
                yield result_title, result_url

    def get_page_content(self, url: str) -> str:
        """
        Get the content of a Fandom page.
        """
        from bs4 import BeautifulSoup
        page_content = self.session.get(url)
        page_content.raise_for_status()
        soup = BeautifulSoup(page_content.text, "html.parser")
        content = soup.find("div", class_="mw-parser-output")
        for superscript_ref in content.find_all("sup", class_="reference"): # type: ignore[union-attr]
            superscript_ref.replace_with("")
        for edit_section in content.find_all("span", class_="mw-editsection"): # type: ignore[union-attr]
            edit_section.replace_with("")
        section_title = None
        for child in content.children: # type: ignore[union-attr]
            if child.name not in {"p", "ul", "ol", "h2", "h3", "dl", "dd", "quote", "table"}: # type: ignore[union-attr]
                child.replace_with("")
            elif child.name in {"h2", "h3"}: # type: ignore[union-attr]
                section_title = re.sub(r"[^a-zA-Z0-9 ]", "", multiline_trim(child.get_text().lower()))
                if section_title in self.ignored_sections:
                    child.replace_with("")
                else:
                    # Add a newline after the section title
                    child.append("\n\n")
            elif section_title in self.ignored_sections:
                child.replace_with("")
            else:
                child.append("\n\n")
        return multiline_trim(content.get_text()) # type: ignore[union-attr]

    def __call__(
        self,
        media_or_franchise: str,
        search: str,
    ) -> str:
        """
        Look up information on Fandom using a search query. Fandom is a wiki site that covers pop culture, including video games, movies, TV shows, and anime.

        :param media_or_franchise: The name of the media or franchise to look up.
        :param search: The key words search for on the Fandom page corresponding to the media or franchise. For best results, search for the name of a character, location, event, or other noun related to the media or franchise.
        :return: The page content as a string.
        """
        if media_or_franchise is None:
            return "No media or franchise provided."
        if search is None:
            return "No search query provided."

        accessed_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        community = self.find_community(media_or_franchise)
        page_summary = None
        # Try to remove the media name from the search query
        search_lower = search.lower()
        media_lower = media_or_franchise.lower()

        search_media_index = search_lower.find(media_lower)
        while search_media_index != -1:
            search = search[:search_media_index] + search[search_media_index + len(media_or_franchise):]
            search_lower = search.lower()
            search_media_index = search_lower.find(media_lower)

        search = search.strip()
        search_lower = search.lower()
        if search_lower.endswith(" wiki"):
            search = search[:-5]
        elif search_lower.endswith(" fandom"):
            search = search[:-7]
        elif search_lower.endswith(" wikipedia"):
            search = search[:-10]
        elif search_lower.endswith(" in"):
            search = search[:-3]

        logger.info(f"Searching for [{media_or_franchise} - {search}] on Fandom")
        for i, (title, url) in enumerate(self.search(search, community=community)):
            self.cite(url, title)
            page_label = f"Result {i+1} for [{media_or_franchise} - {search}]: “{title}” accessed on {accessed_on} at {url}"
            page_content = self.get_page_content(url)
            if page_summary is None:
                page_summary = f"{page_label}\n{page_content}"
            else:
                page_summary += f"\n\n{page_label}\n{page_content}"

        assert page_summary is not None, "No results found."
        return page_summary
