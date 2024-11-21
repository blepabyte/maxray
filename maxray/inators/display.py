"""
Interface types for a Display backend implementation.
"""

from dataclasses import dataclass
from typing import Any
from types import TracebackType


@dataclass
class SetVisible:
    visible: bool


@dataclass
class SetStatus:
    text: str
    style: str = ""


@dataclass
class DumpTraceback:
    exception: BaseException
    traceback: TracebackType


@dataclass
class ShowValue:
    value: Any
    label: str = "show"


@dataclass
class UpdateElement:
    id: str
    contents: Any
    "Up to the backend to figure out a way to display it"


@dataclass
class UpdateStructElement:
    id: str
    contents: dict
    "JSON-like: Implicit ad-hoc interfaces for things like progress bars"


@dataclass
class RemoveElement:
    id: str
