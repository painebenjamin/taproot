__all__ = ["diffusers_is_available"]

def diffusers_is_available() -> bool:
    try:
        import diffusers
        return bool(diffusers)
    except ImportError:
        return False
