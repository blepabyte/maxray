from maxray import NodeContext

from rich.live import Live
from rich.traceback import Traceback, Trace
from rich.pretty import Pretty
from rich.layout import Layout
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Group
from rich.syntax import Syntax
from rich._inspect import Inspect

import inspect
import time
import sys
from contextlib import contextmanager
from pathlib import Path


def render_context(ctx: NodeContext, status_text: str):
    offset_line = (
        ctx.location[0] - ctx.fn_context.line_offset + 3
    )  # TODO: 3 in main script, 2 elsewhere
    code_block = Syntax(
        ctx.fn_context.source,
        "python",
        line_numbers=True,
        line_range=(max(1, offset_line - 5), offset_line + 5),
        highlight_lines={offset_line},
    )

    # TODO: filter stack properly
    text = "\n".join(f.function for f in inspect.stack()[7:])
    current_stack = Text(text)

    header_table = Table(expand=True, title=status_text)

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


class Display:
    """
    Contains state and builds the rich renderable to be shown in the `Live` view.
    """

    def __init__(self, update_ms=500):
        self.live = Live(
            Pretty("Waiting for data to show..."), screen=True, auto_refresh=False
        )
        self.live.start()  # Only start on first use
        self.last_display_tick = time.perf_counter()
        self.update_ms = update_ms
        self.status_text = "[blue]Initialised"
        self.last_ctx = None

        # Arbitrary renderables like progressbars
        self.elements = []

        # Inspect(...)
        self.to_inspect = []

        # Pretty(...)
        self.to_show = []

        # Arbitrary keyed values, shown in a table
        self.tracked = {}

    def inspect(self, x):
        self.to_inspect = [Inspect(x)]
        self.render()

    def __call__(self, *xs):
        self.to_show = [Panel(Group(*(Pretty(x) for x in xs)))]
        self.render()

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

    def clear(self):
        self.to_inspect = []
        self.to_show = []
        self.tracked = {}

    def track(self, keys: dict):
        self.tracked.update(keys)

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

    def render_tracked(self):
        if not self.tracked:
            return []

        table = Table(expand=True, title="Tracked objects")
        table.add_column(
            "Id",
            ratio=1,
        )
        table.add_column("Object", ratio=4)

        # dicts are insertion-ordered: reverse to show newest first (overflow is hidden outside terminal bounds)
        for k, v in reversed(self.tracked.items()):
            table.add_row(
                str(k),
                Panel(Pretty(v), padding=(0, 0)),
            )
        return [table]

    def render(self, *extras):
        root_layout = Layout()

        if self.last_ctx is not None:
            root_layout.split_column(
                Layout(name="header"), main_layout := Layout(name="content")
            )
            root_layout["header"].size = 15
            root_layout["header"].update(
                render_context(self.last_ctx, self.status_text)
            )
        else:
            main_layout = root_layout

        for extra in extras:
            left, right = Layout(), Layout(extra)
            main_layout.split_row(left, right)
            main_layout = left

        elements = [
            *self.elements,
            *self.to_inspect,
            *self.to_show,
            *self.render_tracked(),
        ]
        main_layout.update(Group(*elements))

        self.live.update(root_layout)
        self.live.refresh()

    def update_context(self, ctx: NodeContext):
        self.last_ctx = ctx

    def update_status(self, text: str):
        if text != self.status_text:
            self.status_text = text
            self.render()
