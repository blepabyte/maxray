from __future__ import annotations

import pyarrow as pa
from result import Result, Ok, Err

import inspect
import textwrap
import threading

import uuid
import itertools
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


@dataclass
class Method:
    is_inspect_method: bool
    instance_cls: Optional[type]
    defined_on_cls: Optional[type]

    @staticmethod
    def create(fn):
        is_inspect_method = inspect.ismethod(fn)

        if is_inspect_method:
            if inspect.isclass(fn.__self__):  # classmethod
                self_cls = fn.__self__
            else:
                self_cls = type(fn.__self__)

            return Method(True, self_cls, None)
        return Method(False, None, None)

    def to_dict(self):
        return {
            "instance_cls": self.instance_cls.__name__,
            "defined_on_cls": self.defined_on_cls.__name__,
        }


@dataclass
class FunctionData:
    name: Optional[str]
    qualname: Optional[str]
    module: Optional[str]
    source: Optional[str]
    source_dedent_chars: Optional[int]
    source_file: Optional[str]
    method_info: Method

    id_setattr_ok: Optional[bool] = None

    compile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    compile_triggered_by_fn_id: Optional[str] = None
    compile_trigger_on_fn_line: Optional[int] = None

    @staticmethod
    def create(fn, from_ctx: Optional["NodeContext"]):
        name = getattr(fn, "__name__", None)
        qualname = getattr(fn, "__qualname__", None)
        module = getattr(fn, "__module__", None)

        dedent_chars = None
        try:
            original_source = inspect.getsource(fn)
            # nested functions have excess indentation preventing compile; inspect.cleandoc(source) is an alternative but less reliable
            source = textwrap.dedent(original_source)
            # used to map source back to correct original location
            dedent_chars = original_source.find("\n") - source.find("\n")
        except (OSError, TypeError):
            source = None

        try:
            sourcefile = inspect.getsourcefile(fn)
        except TypeError:
            sourcefile = None

        if from_ctx is not None:
            compile_triggered_by_fn_id = from_ctx.fn_context.compile_id
            compile_trigger_on_fn_line = from_ctx.location[0]
        else:
            compile_triggered_by_fn_id = None
            compile_trigger_on_fn_line = None

        return FunctionData(
            name=name,
            qualname=qualname,
            module=module,
            source=source,
            source_file=sourcefile,
            method_info=Method.create(fn),
            source_dedent_chars=dedent_chars,
            compile_triggered_by_fn_id=compile_triggered_by_fn_id,
            compile_trigger_on_fn_line=compile_trigger_on_fn_line,
        )

    @staticmethod
    def validate(
        fn, from_ctx: Optional["NodeContext"] = None
    ) -> Result[FunctionData, str]:
        # Checks it's at *least* a function of some kind
        if not callable(fn):
            return Err("Not even callable")

        if getattr(fn, "__name__", "") == "<lambda>":
            return Err("<lambda> functions are not supported")

        # Gather all data first for saving
        fd = FunctionData.create(fn, from_ctx)
        if not fd.name:
            reason = "Missing function name"
        elif "<" in fd.name:
            reason = "Invalid character in function name"
        elif not fd.qualname:
            reason = "Missing qualified function name"
        elif not fd.module:
            reason = "Unable to determine function source module"
        elif not fd.source:
            reason = "Unable to find source code of function"
        elif (not fd.source_file) or (not Path(fd.source_file).exists()):
            reason = "Unable to find source file of function"
        elif hasattr(fn, "_MAXRAY_TRANSFORM_ID"):
            reason = (
                f"Function has already been transformed ({fn._MAXRAY_TRANSFORM_ID})"
            )
        else:
            try:
                fn._MAXRAY_TRANSFORM_ID = fd.compile_id
                fd.id_setattr_ok = True
            except AttributeError:
                fd.id_setattr_ok = False

            return Ok(fd)

        FunctionStore.push(ErroredFunction(fd, reason))
        return Err(reason)

    def mark_compiled(self, transformed_fn):
        cf = CompiledFunction(self)
        FunctionStore.push(cf)
        return transformed_fn

    def mark_errored(self, reason: str):
        ef = ErroredFunction(self, reason)
        FunctionStore.push(ef)
        return Err(reason)


@dataclass
class CompiledFunction:
    data: FunctionData

    # Add more fields here with runtime metadata about the function.

    def to_dict(self):
        d = asdict(self.data)
        del d["method_info"]
        d["ok"] = True
        d["error_reason"] = None
        return d


@dataclass
class ErroredFunction:
    data: FunctionData
    error_reason: str

    def to_dict(self):
        d = asdict(self.data)
        del d["method_info"]
        d["ok"] = False
        d["error_reason"] = self.error_reason
        return d


def prepare_function_for_transform(
    fn, from_ctx: Optional["NodeContext"] = None
) -> Result[FunctionData, str]:
    match FunctionData.validate(fn, from_ctx):
        case Ok(fd):
            return Ok(fd)
        case Err(e):
            return Err(e)


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

        if "<" in getattr(fn, "__name__", ""):
            return Err(("Name containing invalid chars", None))
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
            method = Method(True, self_cls, self_cls)
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
    def push(fd: CompiledFunction | ErroredFunction):
        match fd:
            case CompiledFunction():
                FunctionStore.instance.functions[fd.data.compile_id] = fd
            case ErroredFunction():
                FunctionStore.instance.failures[fd.data.compile_id] = fd
            case _:
                raise ValueError(f"{type(fd)}")

    @staticmethod
    def collect():
        # Disable locking by default? (also better context iface)
        with FunctionStore.lock:
            return pa.table(
                pa.array(
                    [
                        fn.to_dict()
                        for fn in itertools.chain(
                            FunctionStore.instance.functions.values(),
                            FunctionStore.instance.failures.values(),
                        )
                    ]
                )
            )


FunctionStore.instance = FunctionStore()
