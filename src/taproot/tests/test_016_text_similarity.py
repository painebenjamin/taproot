from taproot import Task
from taproot.util import debug_logger

def test_text_similarity() -> None:
    """
    Test the text similarity task.
    """
    reference_text = "hello world, how are you?"
    hypothesis_text = "hello world, who are you?"
    jumbled_hypothesis_text = "who you? hello earth"

    with debug_logger() as logger:
        task_class = Task.get("text-similarity", model=None, available_only=False)
        assert task_class is not None
        task = task_class()
        task.load()

        for method in ["bleu", "jaccard", "cosine", "rouge"]:
            similarity = task(
                reference=reference_text,
                hypothesis=hypothesis_text,
                method=method
            )
            logger.info(f"Similarity using {method}: {similarity}")
            assert 0.5 <= similarity <= 1.0

        for method in ["wer", "mer"]:
            error = task(
                reference=reference_text,
                hypothesis=hypothesis_text,
                method=method
            )
            logger.info(f"Error using {method}: {error}")
            assert 0.0 <= error <= 0.5

        for method in ["bleu", "jaccard", "cosine", "rouge"]:
            jumbled_similarity = task(
                reference=reference_text,
                hypothesis=jumbled_hypothesis_text,
                method=method
            )
            logger.info(f"Similarity (jumbled) using {method}: {jumbled_similarity}")
            assert 0.0 <= jumbled_similarity <= 0.5

        for method in ["wer", "mer"]:
            jumbled_error = task(
                reference=reference_text,
                hypothesis=jumbled_hypothesis_text,
                method=method
            )
            logger.info(f"Error (jumbled) using {method}: {jumbled_error}")
            assert 0.5 <= jumbled_error
