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

from loguru import logger


method_wrapper = type(object().__init__)
wrapper_descriptor = type(object.__init__)
builtin_function_or_method = type(locals)


def set_property_on_functionlike(fn, name: str, value: Any, recurse_wrappers=True):
    """
    Tries to mark functions (including class descriptors, wrapped decorators, ...) with a property
    """
    if isinstance(fn, (method_wrapper, wrapper_descriptor, builtin_function_or_method)):
        return False

    ok = True
    if recurse_wrappers and hasattr(fn, "__wrapped__"):
        # Apply at all layers of wrapper
        ok = (
            set_property_on_functionlike(fn.__wrapped__, name, value, True) and ok
        )  # don't short-circuit

    try:
        setattr(fn, name, value)
    except AttributeError as err:
        # Apply to underlying method if bound
        if inspect.ismethod(fn):
            setattr(fn.__func__, name, value)
            return ok
        else:
            ok = False

            logger.error(f"{err}: couldn't set property on {type(fn)}")
    return ok


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

        if not is_inspect_method:
            return Method(False, None, None)

        if inspect.isclass(fn.__self__):  # classmethod
            self_cls = fn.__self__
        else:
            self_cls = type(fn.__self__)

        qual_parts = fn.__qualname__.split(".")
        qual_cls_name = qual_parts[-2] if len(qual_parts) >= 2 else None
        qual_cls = None

        for sup_class in self_cls.mro():
            if sup_class.__name__ == qual_cls_name:
                qual_cls = sup_class
                break

        return Method(True, self_cls, qual_cls)

    def to_dict(self):
        return {
            "is_inspect_method": self.is_inspect_method,
            "instance_cls": self.instance_cls.__name__
            if self.instance_cls is not None
            else None,
            "defined_on_cls": self.defined_on_cls.__name__
            if self.defined_on_cls is not None
            else None,
        }


@dataclass
class FunctionData:
    name: Optional[str]
    qualname: Optional[str]
    module: Optional[str]
    source: Optional[str]
    source_offset_lines: Optional[int]
    source_dedent_chars: Optional[int]
    source_file: Optional[str]
    method_info: Method

    id_setattr_ok: Optional[bool] = None

    compile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    compile_triggered_by_fn_id: Optional[str] = None
    compile_trigger_on_fn_line: Optional[int] = None

    @staticmethod
    def create(fn, from_ctx: Optional["NodeContext"]):
        if hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__

        name = getattr(fn, "__name__", None)
        qualname = getattr(fn, "__qualname__", None)
        module = getattr(fn, "__module__", None)

        dedent_chars = None
        try:
            # inspect.getsource literally calls getsourcelines and joins
            original_source_lines, line_offset = inspect.getsourcelines(fn)
            original_source = "".join(original_source_lines)
            # nested functions have excess indentation preventing compile; inspect.cleandoc(source) is an alternative but less reliable
            source = textwrap.dedent(original_source)
            # used to map source back to correct original location
            dedent_chars = original_source.find("\n") - source.find("\n")
        except (OSError, TypeError):
            source = None
            line_offset = None

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
            source_offset_lines=line_offset,
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
        # Required to be at least a function of some kind to be worth logging errors to FunctionStore
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
            reason = f"Unable to determine function source module: {fd}"
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

    # TODO: Add more fields here with runtime metadata about the function

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
        # TODO: Serialise method info properly
        del d["method_info"]
        d["ok"] = False
        d["error_reason"] = self.error_reason
        return d


def prepare_function_for_transform(
    fn, from_ctx: Optional["NodeContext"] = None
) -> Result[FunctionData, str]:
    return FunctionData.validate(fn, from_ctx)


class FunctionStore:
    instance: FunctionStore
    lock = threading.Lock()

    def __init__(self):
        self.functions = {}
        self.failures = {}

    @staticmethod
    def get(id) -> CompiledFunction | ErroredFunction:
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
        # TODO: Disable locking by default? (design a better context interface?)
        with FunctionStore.lock:
            return pa.table(
                pa.array(
                    [
                        fn.to_dict()
                        for fn in itertools.chain(
                            # RuntimeError: dictionary changed size during iteration
                            list(FunctionStore.instance.functions.values()),
                            list(FunctionStore.instance.failures.values()),
                        )
                    ]
                )
            )


FunctionStore.instance = FunctionStore()
