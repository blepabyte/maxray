from collections import deque
from .base import BaseInator
from .display import (
    DumpTraceback,
    SetStatus,
    SetVisible,
    ShowValue,
    UpdateElement,
    UpdateStructElement,
    RemoveElement,
)
from .core import R, S, Ray
from maxray import NodeContext
from maxray.runner import ExecInfo, RunCompleted, RunErrored, RunAborted
from maxray.function_store import FunctionStore

from rich.live import Live
from rich.traceback import Traceback, Trace
from rich.pretty import Pretty
from rich.progress import Progress
from rich.layout import Layout
from rich.text import Text
from rich.highlighter import ReprHighlighter
from rich.panel import Panel
from rich.table import Table, box
from rich.console import Group
from rich.syntax import Syntax
from rich._inspect import Inspect

from contextlib import contextmanager
import inspect
from pathlib import Path

import time
import sys
import os

from typing import Iterator

from loguru import logger
import click


def render_context(ctx: NodeContext, status_text: str):
    try:
        source_offset_lines = FunctionStore.get(
            ctx.fn_context.compile_id
        ).data.source_offset_lines

    except KeyError:
        source_offset_lines = 0

    ctx_line_in_fn_source = ctx.location[0] - source_offset_lines + 1

    code_block = Syntax(
        ctx.fn_context.source,
        "python",
        line_numbers=True,
        line_range=(max(1, ctx_line_in_fn_source - 5), ctx_line_in_fn_source + 5),
        highlight_lines={ctx_line_in_fn_source},
    )

    # TODO: filter stack properly
    text = "\n".join(f.function for f in inspect.stack()[7:])
    current_stack = Text(text)

    header_table = Table(expand=True, title=status_text, box=box.ROUNDED)

    current_source = ctx.source
    current_file = "/".join(Path(ctx.fn_context.source_file).parts[-3:])

    header_table.add_column(current_file, ratio=2)
    header_table.add_column("Source", ratio=1)
    header_table.add_column("Stack", ratio=2)

    header_table.add_row(
        code_block,
        current_source,
        current_stack,
    )
    return header_table


class Display(BaseInator):
    """
    Builds Rich renderables that are shown in the `Live` view.
    """

    def __init__(self, update_ms: float):
        super().__init__()

        self.live = Live(
            Pretty("Waiting for data to show..."), screen=True, auto_refresh=False
        )
        self.live.start()  # Only start on first use
        self.last_display_tick = time.perf_counter()
        self.update_ms = update_ms
        self.status_text = "[blue]Initialised"
        self.last_ctx = None

        self.last_log_messages = deque(maxlen=10)

        # Arbitrary renderables like progressbars
        self.elements = {}

    @staticmethod
    @click.command()
    @click.option("--interval", type=float, default=500)
    def cli(interval: float):
        return Display(update_ms=interval)

    def iter_display_messages(
        self,
        messages: Iterator[
            SetVisible
            | SetStatus
            | ShowValue
            | DumpTraceback
            | UpdateElement
            | UpdateStructElement
            | RemoveElement
        ],
    ):
        for msg in messages:
            match msg:
                case SetVisible(visible=True):
                    self.live.start()
                    self.render()

                case SetVisible(visible=False):
                    self.live.stop()

                case ShowValue(value=value):
                    self.elements["show_value"] = Pretty(value)
                    self.render()

                case SetStatus(text=text):
                    if text != self.status_text:
                        self.status_text = text
                        self.render()

                case DumpTraceback(exception=exception, traceback=traceback):
                    self.render_traceback(exception, traceback)

                case UpdateElement(id=id, contents=contents):
                    # TODO: handle wider range of things
                    if isinstance(contents, (Progress,)):
                        self.elements[id] = contents

                case RemoveElement(id=id):
                    try:
                        del self.elements[id]
                    except KeyError:
                        ...

            yield

    def xray(self, x, ray: Ray):
        self.last_ctx = ray.ctx
        self.render_maybe()

        match x:
            case RunCompleted():
                self.status_text = "[green]Completed"
                self.render()
            case RunAborted(exception=exception):
                self.status_text = f"[cyan]Aborted ({type(exception).__name__})"
                self.render()
            case RunErrored(exception=exception, traceback=traceback):
                self.status_text = "[red]Errored"
                self.render_traceback(exception, traceback)
            case _:
                set_status = "[yellow]Running..."
                if self.status_text != set_status:
                    self.status_text = set_status
                    self.render()

    def log_message(self, msg):
        match msg.record["extra"]:
            case {
                "maxray_logged_from": logged_from,
                "source_file": source_file,
                "source_line": source_line,
            }:
                source_file = Path(source_file).name
                prefix = f"{logged_from}/{source_file}:{source_line}"
            case _:
                prefix = f"log/{msg.record['file']}:{msg.record['line']}"

        log_level = msg.record["level"].name

        self.last_log_messages.append(f"{log_level}: {msg}")
        self.render()

    @contextmanager
    def enter_session(self, xi: ExecInfo):
        enable_internal_logging = bool(os.environ.get("MAXRAY_LOG_LEVEL"))

        logger.enable("maxray")
        logger.remove()
        logger.add(
            self.log_message,
            filter=lambda record: "maxray_internal" not in record["extra"]
            or enable_internal_logging,
        )

        try:
            with (
                super().enter_session(xi),
                R.DisplayChannel.stack(self.iter_display_messages),
            ):
                yield
        finally:
            self.live.stop()

    @contextmanager
    def hidden(self):
        """
        Pauses and hides live display to avoid cluttering the output with too many frames.
        """
        try:
            self.live.stop()
            yield
            # not in the finally block because display can screw up terminal state on exit
            self.live.start()
        finally:
            self.render()

    def render_maybe(self):
        if (
            tick := time.perf_counter()
        ) - self.last_display_tick > self.update_ms / 1000:
            self.last_display_tick = tick
            self.render()

    def render_traceback(self, exception, traceback):
        trace = Traceback.extract(
            type(exception), exception, traceback, show_locals=True
        )

        traceback = Traceback(
            trace,
            suppress=[sys.modules["maxray"]],
            show_locals=True,
            max_frames=5,
        )

        exc_cause = trace.stacks[0]
        self.status_text = f"[red]PAUSED ON ERROR ({exc_cause.exc_type})\n[orange3]{exc_cause.exc_value})"
        self.render(traceback)

    def render(self, *extras):
        # TODO: don't need to rebuild this every time, can just update components
        root_layout = Layout()

        if self.last_ctx is not None:
            root_layout.split_column(
                Layout(name="header"), main_layout := Layout(name="content")
            )
            # root_layout["header"].size = 20
            root_layout["header"].update(
                render_context(self.last_ctx, self.status_text)
            )
        else:
            main_layout = root_layout

        main_layout.split_column(
            main_layout := Layout(name="element_content"),
            logs_layout := Layout(name="logs_layout"),
        )

        for extra in extras:
            left, right = Layout(), Layout(extra)
            main_layout.split_row(left, right)
            main_layout = left

        main_layout.update(Group(*self.elements.values()))

        logs_layout.update(
            Panel(ReprHighlighter()(Text.from_ansi("\n".join(self.last_log_messages))))
        )

        self.live.update(root_layout)
        self.live.refresh()
