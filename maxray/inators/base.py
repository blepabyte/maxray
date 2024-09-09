from maxray.inators.core import S, Matcher
from maxray.inators.display import Display
from maxray.runner import RunCompleted, RunErrored, RunAborted, Break

import ipdb
import rich
from rich.traceback import Traceback
from rich.live import Live
from rich.pretty import Pretty

import io
import sys
import builtins
from pathlib import Path
from functools import partial
from contextlib import contextmanager
from dataclasses import dataclass

from typing import Any, Optional

import rerun as rr


@dataclass
class LocalContext:
    x: Any
    ctx: Any
    matcher: Optional[Matcher] = None


class BaseInator:
    def __init__(
        self, name: str, rerun: bool, auto_debug: bool, match_assignments: bool
    ):
        """
        Args:
        - name (str): Descriptive name for the program. Only used for visualization and logging.
        - rerun (bool): Whether to auto-init the Rerun visualization library.
        - match_assignments (bool): Enables proper matching on `self.assigned()` by unpacking multiple assignments like `a, b = x`, handling cases where `x` is a stateful iterator or generator by consuming it (converting it to a tuple, throwing away any type information).
        """
        self._name = name
        self._debugger = auto_debug
        self._match_assignments = match_assignments
        self._display = S.define_once("RICH_LIVE_DISPLAY", lambda _: Display())

        if rerun:
            rr.init(self._name, spawn=True)

    def __call__(self, x, ctx):
        if x is builtins.print:
            x = partial(self.print, ctx=ctx)

        if self._match_assignments:
            M = Matcher(x, ctx)
            x = M.unpacked()
            lctx = LocalContext(x, ctx, M)
        else:
            lctx = LocalContext(x, ctx)

        self._last_ctx = lctx
        self._display.update_context(ctx)

        while True:
            try:
                self.xray(x, ctx)
                x = self.maxray(x, ctx)
                break
            except Break:
                self._display.update_status("[violet]PAUSED")
                self.wait_and_reload()
            except Exception as e:
                # Capture and show traceback
                self._display.update_status("[violet]PAUSED")
                self._display.render_traceback(e, e.__traceback__)
                self.wait_and_reload()

        match x:
            case RunCompleted() | RunAborted() | RunErrored():
                self.update_display_state(x)
                self._display.render()
            case _:
                self._display.update_status("[yellow]Running...")
                self._display.render_maybe()
        return x

    def xray(self, x, ctx):
        """
        Override to implement equivalent of @xray
        """
        pass

    def maxray(self, x, ctx):
        """
        Override to implement equivalent of @maxray
        """
        return x

    def runner(self):
        raise NotImplementedError()

    def wait_and_reload(self):
        # Patched in at runtime
        raise NotImplementedError()

    @contextmanager
    def _handle_reload(self):
        """
        Provides control over what happens if an error is encountered while reloading itself.
        """
        try:
            yield
        except Exception as e:
            # Capture and show traceback
            self._display.update_status("[violet]PAUSED")
            self._display.render_traceback(e, e.__traceback__)

    @property
    def display(self) -> Display:
        return self._display

    @contextmanager
    def enter_session(self):
        try:
            yield
        finally:
            self.display.live.stop()

    def update_display_state(self, state: RunCompleted | RunAborted | RunErrored):
        match state:
            case RunCompleted():
                self._display.update_status("[green]Completed")
            case RunAborted(exception=exception):
                self._display.update_status(
                    f"[cyan]Aborted ({type(exception).__name__})"
                )
            case RunErrored(exception=exception, traceback=traceback):
                self._display.update_status("[red]Errored")
                self._display.render_traceback(exception, traceback)

    def print(self, *args, ctx, **kwargs):
        if "file" in kwargs:
            return print(*args, **kwargs)

        print(*args, **kwargs, file=(buf := io.StringIO()))

        source_location = (
            Path(ctx.fn_context.source_file).name + ":" + str(ctx.location[0] + 1)
        )
        rr.log(
            f"print/{source_location}",
            rr.TextLog(buf.getvalue().strip(), level="TRACE"),
        )

    # Utility functions

    def log(self, obj, level="INFO"):
        rr.log("log", rr.TextLog(str(obj), level=level))
        return obj

    def enter_debugger(self, post_mortem: RunErrored | bool = False):
        with self.display.hidden():
            if post_mortem is True:
                # Needs to be an active exception
                ipdb.post_mortem()
            elif isinstance(post_mortem, RunErrored):
                exc_trace = Traceback.extract(
                    type(post_mortem.exception),
                    post_mortem.exception,
                    post_mortem.traceback,
                    show_locals=True,
                )
                traceback = Traceback(
                    exc_trace,
                    suppress=[sys.modules["maxray"]],
                    show_locals=True,
                    max_frames=5,
                )
                rich.print(traceback)
                ipdb.post_mortem(post_mortem.traceback)
            else:
                ipdb.set_trace()

    def assigned(self):
        if self._last_ctx is None:
            raise RuntimeError("Outside of any node context")
        if self._last_ctx.matcher is None:
            raise ValueError("Must enable match_assignments to use `assigned`")

        return self._last_ctx.matcher.assigned()
