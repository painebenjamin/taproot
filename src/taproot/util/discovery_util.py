from __future__ import annotations

from typing import List, Type, Optional, Dict

__all__ = ["DiscoveryMixin", "NamedDiscoveryMixin"]

class DiscoveryMixin:
    """
    A mixin that allows classes to be discovered by introspection.
    """
    include_base_class: bool = False

    @classmethod
    def include_in_enumeration(
        cls,
        introspected_class: Type[DiscoveryMixin]
    ) -> bool:
        """
        :param introspected_class: The class that is being introspected.
        :return: True if the class should be included in the introspection.
        """
        return cls.include_base_class or introspected_class != cls

    @classmethod
    def enumerate(
        cls,
        introspected_class: Optional[Type[DiscoveryMixin]]=None
    ) -> List[Type[DiscoveryMixin]]:
        """
        Enumerate all classes that are subclasses of this class.
        :param introspected_class: The class that is being introspected. If None, the class itself is being introspected.
        :return: A list of classes that are subclasses of this class.
        """
        classes = set()
        if cls.include_in_enumeration(introspected_class or cls):
            classes.add(cls)
        for subclass in cls.__subclasses__():
            if subclass.include_in_enumeration(introspected_class or cls):
                classes.add(subclass)
            classes = classes.union(set(subclass.enumerate(introspected_class or cls)))
        return list(classes)

class NamedDiscoveryMixin(DiscoveryMixin):
    """
    A mixin that allows classes to be discovered by introspection and have a name.
    """
    name: Optional[str] = None
    include_base_class: bool = True

    @classmethod
    def include_in_enumeration(
        cls,
        introspected_class: Type[DiscoveryMixin]
    ) -> bool:
        """
        :param introspected_class: The class that is being introspected.
        :return: True if the class should be included in the introspection.
        """
        return getattr(cls, "name", None) is not None

    @classmethod
    def get(cls, name: str) -> Optional[Type[NamedDiscoveryMixin]]:
        """
        :param name: The name of the class to get.
        :return: The class with the given name, or None if no such class exists.
        """
        for c in cls.enumerate():
            if getattr(c, name, None) == name:
                return c # type: ignore[return-value]
        return None

    @classmethod
    def catalog(cls: Type[DiscoveryMixin]) -> Dict[str, Type[DiscoveryMixin]]:
        """
        :return: A dictionary of classes that are subclasses of this class, indexed by their name.
        """
        return {c.name: c for c in cls.enumerate()} # type: ignore[attr-defined]
