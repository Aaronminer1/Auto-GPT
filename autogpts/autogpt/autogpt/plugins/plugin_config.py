from typing import Any

from pydantic.v1 import BaseModel


class PluginConfig(BaseModel):
    """Class for holding configuration of a single plugin"""

    name: str
    enabled: bool = False
    config: dict[str, Any] = None
