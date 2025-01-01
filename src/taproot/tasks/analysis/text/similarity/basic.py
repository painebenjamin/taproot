from __future__ import annotations

import re

from typing import Dict, List, Optional, Any, TYPE_CHECKING
from typing_extensions import Literal

from collections import Counter
from math import exp

from taproot.constants import *
from taproot.tasks.base import Task

if TYPE_CHECKING:
    import numpy as np

__all__ = ["TextSimilarity"]

class TextSimilarity(Task):
    """
    Uses traditional (non-AI) methods to calculate the similarity between two texts.
    """
    task = "text-similarity"
    default = True
    display_name = "Traditional Text Similarity"

    """Authorship Metadata"""
    author = "Benjamin Paine"
    author_url = "https://github.com/painebenjamin/taproot"
    author_affiliations = ["Taproot"]

    """License Metadata"""
    license = LICENSE_APACHE

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Returns the required packages for this task.
        """
        return {
            "numpy": NUMPY_VERSION_SPEC
        }

    def chunk_text_by_words(self, text: str, chunk_size: int) -> List[str]:
        """
        Chunks the text into chunks of the specified size.

        :param text: The text to chunk.
        :param chunk_size: The size of the chunks.
        :return: The list of chunks.
        """
        words = text.split()
        word_count = len(words)
        chunks = []
        for i in range(0, word_count, chunk_size):
            chunks.append(" ".join(words[i:i+chunk_size]))
        return chunks

    def count_n_grams(self, sentence: str, ngram: int = 4) -> Dict[str, int]:
        """
        Counts the n-grams in the sentence.

        :param sentence: The sentence to count the n-grams in.
        :param ngram: The size of the n-grams to count.
        :return: A dictionary of n-grams and their counts.
        """
        return dict(
            Counter(
                tuple(sentence[i:i+ngram]) # type: ignore[misc]
                for i in range(len(sentence) - ngram + 1)
            )
        )

    def bleu(self, reference: str, hypothesis: str, ngram: int = 4) -> float:
        """
        Calculates the Bilingual Evaluation Understudy (BLEU) score between the reference and hypothesis texts.

        :param reference: The reference text.
        :param hypothesis: The hypothesis text.
        :param ngram: The size of the n-grams to use.
        :return: The BLEU score.
        """
        # Implementation goes here
        reference_words = reference.split()
        hypothesis_words = hypothesis.split()
        num_reference_words = len(reference_words)
        num_hypothesis_words = len(hypothesis_words)
        precision_scores: List[float] = []

        for i in range(1, ngram + 1):
            reference_ngrams = self.count_n_grams(reference, i)
            hypothesis_ngrams = self.count_n_grams(hypothesis, i)

            overlap = sum([
                min(
                    hypothesis_ngrams.get(ngram, 0),
                    reference_ngrams.get(ngram, 0)
                )
                for ngram in hypothesis_ngrams.keys()
            ])

            total_hypothesis_ngrams = sum(hypothesis_ngrams.values())

            precision = overlap / total_hypothesis_ngrams if total_hypothesis_ngrams > 0 else 0.0
            precision_scores.append(precision)

        bleu_score = sum(precision_scores) / ngram
        brevity_penalty = exp(1 - num_reference_words / num_hypothesis_words) if num_hypothesis_words < num_reference_words else 1.0
        return brevity_penalty * bleu_score

    def jaccard(self, reference: str, hypothesis: str) -> float:
        """
        Calculates the Jaccard similarity between the reference and hypothesis texts.

        :param reference: The reference text.
        :param hypothesis: The hypothesis text.
        :return: The Jaccard similarity.
        """
        reference_words = set(reference.split())
        hypothesis_words = set(hypothesis.split())
        intersection = reference_words.intersection(hypothesis_words)
        union = reference_words.union(hypothesis_words)
        return len(intersection) / len(union)

    def cosine(self, reference: str, hypothesis: str) -> float:
        """
        Calculates the cosine similarity between the reference and hypothesis texts.

        :param reference: The reference text.
        :param hypothesis: The hypothesis text.
        :return: The cosine similarity.
        """
        reference_words = reference.split()
        hypothesis_words = hypothesis.split()
        reference_counts = Counter(reference_words)
        hypothesis_counts = Counter(hypothesis_words)
        intersection = set(reference_words).intersection(set(hypothesis_words))
        numerator = sum(reference_counts[word] * hypothesis_counts[word] for word in intersection)
        denominator = sum(reference_counts[word] ** 2 for word in reference_words) ** 0.5 * sum(hypothesis_counts[word] ** 2 for word in hypothesis_words) ** 0.5
        return numerator / denominator # type: ignore[no-any-return]

    def get_distance_matrix(self, reference: str, hypothesis: str) -> np.ndarray[Any, Any]:
        """
        Calculates the distance matrix between the reference and hypothesis texts.

        Shared between WER and MER.

        :param reference: The reference text.
        :param hypothesis: The hypothesis text.
        :return: The distance matrix.
        """
        import numpy as np
        reference_words = reference.split()
        hypothesis_words = hypothesis.split()

        num_reference_words = len(reference_words)
        num_hypothesis_words = len(hypothesis_words)

        distance = np.zeros((num_reference_words + 1, num_hypothesis_words + 1))

        # Initialize the distance matrix
        for i in range(num_reference_words + 1):
            distance[i][0] = i
        for j in range(num_hypothesis_words + 1):
            distance[0][j] = j

        # Compute the distance matrix
        for i in range(1, num_reference_words + 1):
            for j in range(1, num_hypothesis_words + 1):
                if reference_words[i - 1] == hypothesis_words[j - 1]:
                    distance[i][j] = distance[i - 1][j - 1]
                else:
                    distance[i][j] = min(
                        distance[i - 1][j] + 1,    # Deletion
                        distance[i][j - 1] + 1,    # Insertion
                        distance[i - 1][j - 1] + 1 # Substitution
                    )

        return distance

    def rouge(self, reference: str, hypothesis: str, ngram: int = 2) -> float:
        """
        Calculates the Recall Oriented Understudy for Gisting Evaluation (ROUGE) score
        between the reference and hypothesis texts.

        :param reference: The reference text.
        :param hypothesis: The hypothesis text.
        :param ngram: The size of the n-grams to use.
        :return: The ROUGE score.
        """
        reference_words = reference.split()
        hypothesis_words = hypothesis.split()

        reference_ngrams = self.count_n_grams(reference, ngram)
        hypothesis_ngrams = self.count_n_grams(hypothesis, ngram)

        overlap = sum([
            min(
                hypothesis_ngrams.get(ngram, 0),
                reference_ngrams.get(ngram, 0)
            )
            for ngram in hypothesis_ngrams.keys()
        ])
        total_reference_ngrams = sum(reference_ngrams.values())

        recall = overlap / total_reference_ngrams if total_reference_ngrams > 0 else 0.0
        return recall

    def wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculates the Word Error Rate (WER) between the reference and hypothesis texts.

        :param reference: The reference text.
        :param hypothesis: The hypothesis text.
        :return: The WER.
        """
        distance = self.get_distance_matrix(reference, hypothesis)
        num_reference_words, num_hypothesis_words = distance.shape
        num_reference_words -= 1
        num_hypothesis_words -= 1
        num_substitutions = distance[num_reference_words][num_hypothesis_words]
        wer = num_substitutions / num_reference_words
        return wer # type: ignore[no-any-return]

    def mer(self, reference: str, hypothesis: str) -> float:
        """
        Calculates the Match Error Rate (MER) between the reference and hypothesis texts.

        :param reference: The reference text.
        :param hypothesis: The hypothesis text.
        :return: The MER.
        """
        distance = self.get_distance_matrix(reference, hypothesis)
        num_reference_words, num_hypothesis_words = distance.shape
        num_reference_words -= 1
        num_hypothesis_words -= 1
        num_substitutions = distance[num_reference_words][num_hypothesis_words]
        mer = num_substitutions / num_hypothesis_words
        return mer # type: ignore[no-any-return]

    def __call__( # type: ignore[override]
        self,
        *,
        reference: str,
        hypothesis: str,
        method: Literal["bleu", "jaccard", "cosine", "rouge", "wer", "mer", "chunked-wer", "chunked-mer"] = "bleu",
        ngram: Optional[int] = None,
        chunk_size: int = 100,
        case_sensitive: bool = False,
    ) -> float:
        """
        Calculates the similarity between the reference and hypothesis texts.

        :param reference: The reference text.
        :param hypothesis: The hypothesis text.
        :param method: The method to use for calculating the similarity.
        :param ngram: The size of the n-grams to use.
        :param chunk_size: The size of the chunks to use for chunked methods.
        :param case_sensitive: Whether to consider case sensitivity.
        :return: The similarity score.
        """
        # Remove non-word characters
        reference = re.sub(r"[^\w\s]", "", reference)
        hypothesis = re.sub(r"[^\w\s]", "", hypothesis)

        # Remove newlines
        reference = reference.replace("\n", " ")
        hypothesis = hypothesis.replace("\n", " ")

        # Remove extra spaces
        reference = re.sub(r"\s+", " ", reference)
        hypothesis = re.sub(r"\s+", " ", hypothesis)

        # If not case sensitive, make lower
        if not case_sensitive:
            reference = reference.lower()
            hypothesis = hypothesis.lower()

        # Execute the method
        if method == "bleu":
            return self.bleu(reference, hypothesis, ngram=4 if ngram is None else ngram)
        elif method == "jaccard":
            return self.jaccard(reference, hypothesis)
        elif method == "cosine":
            return self.cosine(reference, hypothesis)
        elif method == "rouge":
            return self.rouge(reference, hypothesis, ngram=2 if ngram is None else ngram)
        elif method == "wer":
            return self.wer(reference, hypothesis)
        elif method == "mer":
            return self.mer(reference, hypothesis)
        raise ValueError(f"Invalid method: {method}")
