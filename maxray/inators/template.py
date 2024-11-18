"""
A composable layer that can be applied on top of the execution of any Python script or module.
"""

from maxray.inators.core import R, S, NodeContext, Ray
from maxray.inators.base import BaseInator
from maxray.runner import (
    MAIN_FN_NAME,
    ExecInfo,
    RunAborted,
    RunCompleted,
    RunErrored,
    AbortRun,
    RestartRun,
    Break,
)

from contextlib import contextmanager

import click
import rerun as rr


class Inator(BaseInator):
    def __init__(self):
        super().__init__()

    @staticmethod
    @click.command()
    def cli():
        return Inator()

    def xray(self, x, ray: Ray):
        """
        (x)ray: a callback for each evaluated expression corresponding to an AST node in the original source code.

        Args:
            `x`: the "current" value, evaluated sequentially before/after other `BaseInator`s that have been applied.
            `ray`: provides access to the evaluation context; local variables, source code, and utility functions to match on function calls, variable assignments, `with` entries, etc.
        """
        ctx: NodeContext = ray.ctx

    def maxray(self, x, ray: Ray):
        """
        (m)utable xray: this function must return a value to replace `x` in the evaluation of the source expression.

        Coarse control flow can be achieved via `exit()` or raising one of `Break()`, `AbortRun()`, or `RestartRun()`.
        """
        ctx: NodeContext = ray.ctx

        match ctx.source:
            case "...":
                ...

        match x:
            case _:
                ...

        match ray.locals():
            case {}:
                ...

        match ray.assigned():
            case {"df": df}:
                ...
            case _:
                ...

        return x

    @contextmanager
    def enter_session(self, xi: ExecInfo):
        try:
            with super().enter_session(xi):
                yield
        finally:
            ...

    def runner(self):
        """
        Generator interface that lets you run the original program arbitrarily many times, handling error cases and cleanup logic.
        Each iteration yields another run of the program.

        Note:
            Changes to this function are NOT applied on reload -- modify `match_run_result` below instead
        """
        try:
            while True:
                result: RunCompleted | RunAborted | RunErrored = yield
                match result:
                    case RunAborted(exception=RestartRun()):
                        continue
                    case RunAborted(exception=AbortRun()):
                        ...
                    case RunAborted():
                        return

                self.match_run_result(result)

        finally:
            ...

    def match_run_result(self, result: RunCompleted | RunAborted | RunErrored):
        match result:
            case RunCompleted():
                ...
            case RunErrored():
                with self.hide_display():
                    result.show_traceback()

        self.wait_and_reload()
