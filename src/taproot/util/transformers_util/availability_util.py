__all__ = ["transformers_is_available"]

def transformers_is_available() -> bool:
    try:
        import transformers # type: ignore[import-untyped,import-not-found,unused-ignore]
        return bool(transformers)
    except ImportError:
        return False
