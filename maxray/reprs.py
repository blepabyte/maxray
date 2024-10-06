import numpy as np

from collections.abc import Mapping
import sys

from loguru import logger


def type_name(x):
    try:
        return type(x).__name__
    except Exception:
        return "{unrepresentable type}"


def structured_repr(x):
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

        return type_name(x)
    except Exception as e:
        logger.exception(e)
        return "<error_cannot_repr>"
