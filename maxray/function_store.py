from __future__ import annotations

import pyarrow as pa
from result import Result, Ok, Err

import inspect
import textwrap
import threading

import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


def set_property_on_functionlike(fn, name: str, value: Any):
    """
    Tries to apply to functions, class descriptors, wrapped decorators, etc.
    """
    raise NotImplementedError()


def get_fn_name(fn):
    """
    Get a printable representation of the function for human-readable errors
    """
    if hasattr(fn, "__name__"):
        name = fn.__name__
    else:
        try:
            name = repr(fn)
        except Exception:
            name = f"<unrepresentable function of type {type(fn)}>"

    return f"{name} @ {id(fn)}"


# TODO: fix typing interface
@dataclass
class FailedWithFunction:
    name: Optional[str]
    qualname: Optional[str]
    module: Optional[str]
    source: Optional[str]
    source_file: Optional[str]

    compile_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Method:
    instance_cls: type
    defined_on_cls: type

    def to_dict(self):
        return {
            "instance_cls": self.instance_cls.__name__,
            "defined_on_cls": self.defined_on_cls.__name__,
        }


@dataclass
class WithFunction:
    name: str
    qualname: str
    module: str
    source: str
    source_file: str

    method: Method | None

    compile_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    added_context: dict = field(default_factory=dict)
    "Arbitrary extra data that won't be saved to the output table"

    @staticmethod
    def try_for_transform(
        fn,
    ) -> Result[tuple[Any, WithFunction], tuple[str, WithFunction | None]]:
        """
        Does basic validation and gathers + stores metadata as a precondition for `recompile_fn_with_transform` to run.
        Also unnests decorator/descriptor wrappers to get the target function to apply the transform to.
        """
        if not callable(fn):
            return Err(("Not even callable", None))

        # Unwraps things like functools.wraps, functools.lru_cache as well as descriptors like @classmethod and @staticmethod to get to the underlying function
        wrappers = []
        if hasattr(fn, "__wrapped__"):
            wrappers.append(fn)
            fn = fn.__wrapped__

        if "<" in getattr(fn, "__name__", ""):
            return Err(("Name containing invalid chars", None))

        # Determines if we treat this as a method
        is_method = inspect.ismethod(fn)
        if is_method:
            if inspect.isclass(fn.__self__):  # classmethod
                self_cls = fn.__self__
            else:
                self_cls = type(fn.__self__)

            use_name = getattr(fn, "__name__", "")
            if use_name.startswith("__") and not use_name.endswith(
                "__"
            ):  # mangled method
                use_name = f"_{self_cls.__name__}{use_name}"

            # try:
            #     defn_cls_name, _name = fn.__qualname__.rsplit(".", 1)
            #     (defn_cls,) = [
            #         q for q in self_cls.mro() if q.__qualname__ == defn_cls_name
            #     ]
            #     fn = defn_cls.__dict__[use_name]
            # except Exception as e:
            #     # print(e)
            #     # breakpoint()
            #     defn_cls = self_cls

            # last_fn = None
            # while hasattr(fn, "__wrapped__") or inspect.ismethod(fn):
            #     if last_fn is fn:
            #         raise RecursionError("The method won't fucking die")
            #     last_fn = fn

            #     while hasattr(fn, "__wrapped__"):
            #         wrappers.append(fn)
            #         fn = fn.__wrapped__

            #     # This is a descriptor, let's get the actual function object
            #     # for defn_cls in self_cls.mro():
            #     #     # TODO: use_name or plain name? is mangling done on dict?
            #     #     if use_name in defn_cls.__dict__:
            #     #         fn = defn_cls.__dict__[use_name]
            #     #         assert not inspect.ismethod(fn)
            #     #         break

            #     while hasattr(fn, "__wrapped__"):
            #         wrappers.append(fn)
            #         fn = fn.__wrapped__

            # print([(type(w), w) for w in wrappers])
            # while hasattr(fn, "__wrapped__"):
            #     wrappers.append(fn)
            #     fn = fn.__wrapped__
        else:
            self_cls = None
            defn_cls = None

        if not callable(fn):
            return Err(("Resolved object from fn not callable", None))

        name = getattr(fn, "__name__", None)
        qualname = getattr(fn, "__qualname__", None)
        module = getattr(fn, "__module__", None)

        try:
            original_source = inspect.getsource(fn)
            # nested functions have excess indentation preventing compile; inspect.cleandoc(source) is an alternative but less reliable
            source = textwrap.dedent(original_source)
            # used to map source back to correct original location
            dedent_chars = original_source.find("\n") - source.find("\n")
        except OSError:
            source = None
        except TypeError:
            source = None

        try:
            sourcefile = inspect.getsourcefile(fn)
        except TypeError:
            sourcefile = None

        if is_method:
            method = Method(self_cls, self_cls)
        else:
            method = None
        maybe_fn = WithFunction(name, qualname, module, source, sourcefile, method)

        if source is None:
            # assert not hasattr(fn, "_MAXRAY_TRANSFORM_ID")
            return maybe_fn.errored(f"No source code for function {get_fn_name(fn)}")
        maybe_fn.added_context["dedent_chars"] = dedent_chars

        # codegen (e.g. numpy's array hooks) results in functions that have source code but no corresponding source file
        # the source file of `np.unique` is <__array_function__ internals>
        if sourcefile is None or not Path(sourcefile).exists():
            return maybe_fn.errored(
                f"Non-existent source file ({sourcefile}) for function {get_fn_name(fn)}"
            )

        if name is None:
            return maybe_fn.errored(
                f"There is no __name__ for function {get_fn_name(fn)}"
            )

        if name == "<lambda>":
            return maybe_fn.errored("Cannot safely recompile lambda functions")

        if qualname is None:
            return maybe_fn.errored(
                f"There is no __qualname__ for function {get_fn_name(fn)}"
            )

        if module is None:
            return maybe_fn.errored(
                f"There is no __module__ for function {get_fn_name(fn)}"
            )

        # Is there a better way to "track" function identity? Not sure about weakref...
        try:
            fn._MAXRAY_TRANSFORM_ID = maybe_fn.compile_id
        except AttributeError as err:
            # print(err)
            pass
            # breakpoint()

        return Ok((fn, maybe_fn))

    def compiled(self):
        # if (instance := FunctionStore.instance.get(None)) is not None:
        with FunctionStore.lock:
            FunctionStore.instance.functions[self.compile_id] = self
        return self

    def errored(self, err):
        with FunctionStore.lock:
            FunctionStore.instance.failures[self.compile_id] = self
        return Err((err, self))

    def to_dict(self):
        data = asdict(self)
        if self.method is not None:
            data["method"] = self.method.to_dict()
        return data


class FunctionStore:
    instance: FunctionStore
    lock = threading.Lock()

    def __init__(self):
        self.functions = {}
        self.failures = {}

    @staticmethod
    def get(id) -> WithFunction:
        return FunctionStore.instance.functions[id]

    @staticmethod
    def collect():
        # Disable locking by default? (also better context iface)
        with FunctionStore.lock:
            return pa.table(
                pa.array(
                    [fn.to_dict() for fn in FunctionStore.instance.functions.values()]
                )
            )


FunctionStore.instance = FunctionStore()
