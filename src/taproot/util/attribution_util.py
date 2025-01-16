from typing import List, Optional
from taproot.constants import *

__all__ = ["AttributionMixin"]

class AttributionMixin:
    """
    Mixin class for authorship and licensing metadata.
    """

    """Authorship metadata"""
    author: Optional[str] = None # Author of the components needed for the task
    author_url: Optional[str] = None # URL to author's website or profile
    author_additional: Optional[List[str]] = None # Additional authors of the components needed for the task
    author_affiliations: Optional[List[str]] = None # Affiliations of the authors of the components needed for the task
    author_journal: Optional[str] = None # Journal or conference where the components needed for the task were published
    author_journal_pages: Optional[str] = None # Pages in the journal or conference where the components needed for the task were published
    author_journal_volume: Optional[str] = None # Volume of the journal or conference where the components needed for the task were published
    author_journal_title: Optional[str] = None # Title of the publication in the journal or conference where the components needed for the task were published
    author_journal_year: Optional[int] = None # Year of publication of the components needed for the task
    finetune_author: Optional[str] = None # Author of the finetuned model
    finetune_author_url: Optional[str] = None # URL to author's website or profile
    implementation_author: Optional[str] = None # Author of the implementation if different from paper/journal authors
    implementation_author_url: Optional[str] = None # URL to author's website or profile

    """License metadata"""
    license: Optional[str] = None  # License for the components needed for the task
    license_url: Optional[str] = None  # URL to license details if not a standard license
    license_attribution: Optional[bool] = None  # Whether the license requires attribution
    license_derivatives: Optional[bool] = None  # Whether the license allows derivatives
    license_redistribution: Optional[bool] = None  # Whether the license allows redistribution
    license_copy_left: Optional[bool] = None  # Whether the license is a copyleft license
    license_commercial: Optional[bool] = None  # Whether the license allows commercial use
    license_hosting: Optional[bool] = None  # Whether the license allows hosting

    @classmethod
    def get_author_citation(cls) -> str:
        """
        Get the author citation.
        """
        author_string = ""

        if cls.author:
            author_string += cls.author
            if cls.author_additional is not None:
                if len(cls.author_additional) == 1:
                   author_string += f"and {cls.author_additional[0]}"
                else:
                    author_string += f", {', '.join(cls.author_additional[:-1])} and {cls.author_additional[-1]}"

        if cls.author_affiliations:
            if author_string:
                author_string += "\n"
            if len(cls.author_affiliations) == 1:
                author_string += f"{cls.author_affiliations[0]}"
            else:
                author_string += f"{', '.join(cls.author_affiliations[:-1])} and {cls.author_affiliations[-1]}"

        if cls.author_journal:
            if author_string:
                author_string += "\n"
            author_string += f"Published in {cls.author_journal}"
            if cls.author_journal_volume:
                author_string += f", vol. {cls.author_journal_volume}"
            if cls.author_journal_pages:
                author_string += f", pp. {cls.author_journal_pages}"
            if cls.author_journal_title:
                author_string += f", “{cls.author_journal_title}”"
            if cls.author_journal_year:
                author_string += f", {cls.author_journal_year}"

        if cls.author_url:
            if author_string:
                author_string += "\n"
            author_string += cls.author_url

        if cls.finetune_author:
            if author_string:
                author_string += "\n"
            author_string += f"Finetuned by {cls.finetune_author}"
            if cls.finetune_author_url:
                author_string += f" ({cls.finetune_author_url})"

        if cls.implementation_author:
            if author_string:
                author_string += "\n"
            author_string += f"Implementation by {cls.implementation_author}"
            if cls.implementation_author_url:
                author_string += f" ({cls.implementation_author_url})"

        return author_string

    @classmethod
    def get_license_citation(cls) -> str:
        """
        Get the license citation.
        """
        license_string = ""

        if cls.license:
            if cls.license in LICENSE_NAMES:
                license_string += LICENSE_NAMES[cls.license]
            else:
                license_string += cls.license
            if cls.license_url:
                license_string += f" ({cls.license_url})"
            elif cls.license in LICENSE_URLS:
                license_string += f" ({LICENSE_URLS[cls.license]})"
        return license_string

    @classmethod
    def get_license_allowances(
        cls,
        allowed_character: str="✅",
        disallowed_character: str="❌",
        delimiter: str="\n",
    ) -> str:
        """
        Get the license allowances.
        """
        if cls.license and cls.license in LICENSE_ALLOWANCES:
            (
                default_attribution,
                default_derivatives,
                default_redistribution,
                default_copy_left,
                default_commercial,
                default_hosting
            ) = LICENSE_ALLOWANCES[cls.license]
        else:
            default_attribution = True
            default_derivatives = False
            default_redistribution = True # This is true for all models we distribute by their nature
            default_copy_left = False
            default_commercial = False
            default_hosting = False

        attribution = cls.license_attribution if cls.license_attribution is not None else default_attribution
        derivatives = cls.license_derivatives if cls.license_derivatives is not None else default_derivatives
        redistribution = cls.license_redistribution if cls.license_redistribution is not None else default_redistribution
        copy_left = cls.license_copy_left if cls.license_copy_left is not None else default_copy_left
        commercial = cls.license_commercial if cls.license_commercial is not None else default_commercial
        hosting = cls.license_hosting if cls.license_hosting is not None else default_hosting

        return delimiter.join([
            f"{allowed_character if attribution else disallowed_character} Attribution Required",
            f"{allowed_character if derivatives else disallowed_character} Derivatives Allowed",
            f"{allowed_character if redistribution else disallowed_character} Redistribution Allowed",
            f"{allowed_character if copy_left else disallowed_character} Copyleft (Share-Alike) Required",
            f"{allowed_character if commercial else disallowed_character} Commercial Use Allowed",
            f"{allowed_character if hosting else disallowed_character} Hosting Allowed",
        ])
