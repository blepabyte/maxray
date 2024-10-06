from __future__ import annotations

from .function_store import CompiledFunction, FunctionStore
import inspect
from dataclasses import dataclass, field
from contextvars import ContextVar
from pathlib import Path

from typing import Any, Callable, Optional

from .logging import logger


@dataclass
class FnContext:
    impl_fn: Callable
    name: str  # qualname
    module: str
    source: str
    source_file: str
    line_offset: int
    "Line number in `source_file` at which the function definition begins."

    compile_id: str

    def __repr__(self):
        # Call count not included in repr so the same source location can be "grouped by" over multiple calls
        return f"{self.module}/{self.name}"


@dataclass
class NodeContext:
    id: str
    """
    Identifier for the type of syntax node this event came from (`name/x`, `call/foo`, ...)
    """

    source: str

    fn_context: FnContext
    """
    Resolved info of the function within which the source code of this node was transformed/compiled.
    
    Notes:
        Not necessarily the actual active/called function (w.r.t. call stack) in the case of nested function defs (transformed as a whole).
    """

    location: tuple[int, int, int, int]
    """
    (start_line, end_line, start_col, end_col) as 0-indexed location of `source` in `fn_context.source_file`

    Notes:
        end_col is the exclusive endpoint of the range
    """

    local_scope: Optional[dict] = None
    "When `pass_scope` is True, contains the output of `builtins.locals()` evaluated in the scope of the source expression"

    props: dict = field(default_factory=lambda: {})

    def __repr__(self):
        return f"{self.fn_context}/{self.id}"

    def to_dict(self):
        return {
            "id": self.id,
            "source": self.source,
            "location": self.location,
            "fn_context": {
                "name": self.fn_context.name,
                "module": self.fn_context.module,
                "source_file": self.fn_context.source_file,
                "call_count": self.fn_context.call_count.get(),
            },
            # TODO: include props
        }

    def _set_assigned(self, targets: list[str]):
        self.props["assigned"] = {"targets": targets}
        return self

    def _set_iterated(self, target: str):
        self.props["iterated"] = {"target": target}
        return self

    def _set_returned(self, value_source: str):
        self.props["returned"] = {"value_source": value_source}
        return self

    def _set_called(self, call_args, call_kwargs):
        self.props["called"] = {"args": call_args, "kwargs": call_kwargs}
        return self

    def _set_entered(self, source, as_var):
        self.props["entered"] = {"source": source}
        if as_var is not None:
            self.props["entered"]["as_var"] = as_var
        return self


class RayContext:
    """
    Captures the state of a point (syntax node) in the source code of the original program.

    One instance is created for each point in the program, that is then passed to multiple handlers.
    """

    def __init__(self, x, ctx: NodeContext, *, unpack_assignments: bool = False):
        self._unpack_assignments = unpack_assignments
        if self._unpack_assignments:
            self._x, self._assigned = self._unpack_assign_context(x, ctx)
        else:
            self._x = x
            self._assigned = {}
        self.ctx = ctx

    @staticmethod
    def _unpack_assign_context(x, ctx):
        match ctx.props:
            case {"assigned": {"targets": targets}}:
                if len(targets) > 1:
                    if inspect.isgenerator(x) or isinstance(x, (map, filter)):
                        # Greedily consume iterators before assignment
                        unpacked_x = tuple(iter(x))
                    else:
                        # Otherwise for chained equality like a = b, c = it, code may rely on `a` being of the original type
                        unpacked_x = x

                    # TODO: doesn't work for starred assignments: x, *y, z = iterable
                    assigned = {target: val for target, val in zip(targets, unpacked_x)}
                    return unpacked_x, assigned

                elif len(targets) == 1:
                    return x, {targets[0]: x}
                else:
                    return x, {}
            case _:
                return x, {}

    def __repr__(self):
        # path_parts = Path(self.ctx.fn_context.source_file).resolve().parts[-4:]
        return f"""Ray {{
    {self.ctx.fn_context.name}
    {self.ctx.fn_context.source_file}:{self.ctx.location[0]}
}}"""

    def value(self):
        return self._x

    def locals(self):
        match self.ctx.local_scope:
            case dict():
                return self.ctx.local_scope
            case None:
                return {}
            case _:
                raise TypeError(
                    f"Unexpected type {type(self.ctx.local_scope)} for local scope"
                )

    def assigned(self):
        if not self._unpack_assignments:
            logger.warning(
                "Matching on .assigned() but assignment unpacking was not enabled"
            )
        return self._assigned

    def iterated(self):
        match self.ctx.props:
            case {"iterated": {"target": target}}:
                return [target]
            case _:
                return []

    def returned(self): ...

    def called(self):
        """
        For `f(x)`, will match on evaluation of `f`.

        Allows immediate substitution with a wrapping callable before invocation.
        """
        match self.ctx.props:
            case {"called": {"args": args, "kwargs": kwargs}}:
                called = {"target": self.ctx.source, "args": args, "kwargs": kwargs}
                try:
                    compile_id = self.value().__getattribute__("_MAXRAY_TRANSFORM_ID")
                    # TODO: could lookup in function store?
                    called["fn_compile_id"] = compile_id

                except AttributeError:
                    ...
                except TypeError:
                    ...
                return called
            case _:
                return {}

    def entered(self):
        """
        Returns:
            {
                source: Source code of the LHS being entered
                as_var: Variable binding after "as", if present
            }
        """
        match self.ctx.props:
            case {"entered": entered}:
                return entered
            case _:
                return {}

    def transformed(self, id=None):
        if id is not None:
            transform_id = id
        else:
            try:
                transform_id = self.value().__getattribute__("_MAXRAY_TRANSFORM_ID")
            except Exception:
                return {}
        fn = FunctionStore.get(transform_id)
        return {
            "id": transform_id,
            "compiled": isinstance(fn, CompiledFunction),
            "data": fn.data,
        }
