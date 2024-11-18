from __future__ import annotations

import numpy as np
from result import Ok, Err, Result

from dataclasses import dataclass
from collections.abc import Mapping
import sys
from typing import Any

from loguru import logger


def type_name(x):
    try:
        return type(x).__name__
    except Exception:
        return "{unrepresentable type}"


def function_name(f):
    try:
        return f.__name__
    except Exception:
        return "{unrepresentable function}"


def structured_repr(x, terminal_repr=type_name):
    try:
        match x:
            case np.ndarray():
                size = "x".join(str(dim) for dim in x.shape)
                return f"[{size}]"

            case list():
                if x:
                    return f"list[{structured_repr(x[0])} x {len(x)}]"

                else:
                    return "list[]"

            case tuple():
                if len(x) <= 3:
                    el_reprs = ", ".join(structured_repr(el) for el in x)
                else:
                    el_reprs = ", ".join(structured_repr(el) for el in x[:3]) + ", ..."
                return f"({el_reprs})"

            case bool():
                return str(x)

            case Mapping() if (keys := list(x)) and isinstance(keys[0], str):
                inner_repr = ", ".join(f"{k}: {structured_repr(x[k])}" for k in keys)
                return f"{type_name(x)} {{{inner_repr}}}"

        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            if isinstance(x, torch.Tensor):
                size = "x".join(str(dim) for dim in x.shape)
                return f"[{size}]"

        if "awkward" in sys.modules:
            ak = sys.modules["awkward"]
            if isinstance(x, ak.Array):
                return f"[{str(x.type)}]"

    except Exception as e:
        logger.exception(e)

    return terminal_repr(x)


def literal_repr(x):
    match x:
        case str():
            return x
        case _:
            return repr(x)


def partial_structured_repr(x, terminal_repr, levels: int) -> list:
    if levels == 0:
        return ["..."]

    try:
        match x:
            case np.ndarray():
                size = "x".join(str(dim) for dim in x.shape)
                return [f"[{size}]"]

            case list():
                if x:
                    return [
                        "list [",
                        *partial_structured_repr(x[0], terminal_repr, levels - 1),
                        f" x {len(x)}]",
                    ]

                else:
                    return ["list []"]

            case tuple():
                if len(x) == 0:
                    return ["()"]

                segments = ["("]
                for item in x:
                    segments.extend(
                        partial_structured_repr(item, terminal_repr, levels - 1)
                    )
                    segments.append(", ")

                if segments[-1] == ", ":
                    segments.pop()

                return segments + [")"]

            case bool():
                return [str(x)]

            case Mapping() if (keys := list(x)) and isinstance(keys[0], str):
                if (tn := type_name(x)) != "dict":
                    segments = [type_name(x), " {"]
                else:
                    segments = ["{"]

                count = 0
                for k in keys:
                    if count > 10:
                        segments.append("...")
                        break

                    v = x[k]
                    segments.append(literal_repr(k))
                    segments.append(": ")
                    segments.extend(
                        partial_structured_repr(v, terminal_repr, levels - 1)
                    )
                    segments.append(", ")

                    count += 1

                if segments[-1] == ", ":
                    segments.pop()

                segments.append("}")
                return segments

            # TODO: dataclasses + attrs?

        # these are not structural reprs
        # if "torch" in sys.modules:
        #     torch = sys.modules["torch"]
        #     if isinstance(x, torch.Tensor):
        #         size = "x".join(str(dim) for dim in x.shape)
        #         return [f"[{size}]"]

        # if "awkward" in sys.modules:
        #     ak = sys.modules["awkward"]
        #     if isinstance(x, ak.Array):
        #         return [f"[{str(x.type)}]"]

    except Exception as e:
        logger.exception(e)

    return [terminal_repr(x)]
