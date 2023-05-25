from typing import Any


class Factory:
    def __init__(self):
        self._objects = {}

    def register_object(self, key: str, model: Any):
        """
        Args:
            key:
            model:

        Returns:
        
        """
        self._objects[key] = model

    def create(self, key: str, **kwargs):
        """
        Args:
            key:
            kwargs:
            
        Returns:

        """
        obj = self._objects.get(key)
        if not obj:
            raise ValueError(key)
        return obj(**kwargs)


class ObjectProvider(Factory):
    def get(self, idx: str, **kwargs):
        return self.create(idx, **kwargs)
