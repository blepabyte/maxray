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

import ipdb

from contextlib import contextmanager
from types import TracebackType

import click
from loguru import logger
import rerun as rr


class IPDB(BaseInator):
    def __init__(self):
        super().__init__()

    @staticmethod
    @click.command()
    def cli():
        return IPDB()

    def xray(self, x, ray: Ray):
        """
        (x)ray: a callback for each evaluated expression corresponding to an AST node in the original source code.

        Args:
            `x`: the "current" value, evaluated sequentially before/after other `BaseInator`s that have been applied.
            `ray`: provides access to the evaluation context; local variables, source code, and utility functions to match on function calls, variable assignments, `with` entries, etc.
        """
        match x:
            case RunErrored():
                with self.hide_display():
                    x.show_traceback()
                    self.enter_debugger(post_mortem=x.traceback)

    def enter_debugger(self, post_mortem: TracebackType | bool = False):
        if post_mortem is True:
            # Needs to be an active exception
            ipdb.post_mortem()
        elif isinstance(post_mortem, TracebackType):
            ipdb.post_mortem(post_mortem)
        else:
            ipdb.set_trace()

    @contextmanager
    def enter_session(self, xi: ExecInfo):
        try:
            with super().enter_session(xi):
                yield
        finally:
            ...
