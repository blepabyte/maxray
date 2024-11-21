from .display import (
    DumpTraceback,
    SetStatus,
    SetVisible,
    ShowValue,
    UpdateElement,
    UpdateStructElement,
    RemoveElement,
)

from maxray.inators.core import R, S, NodeContext, Ray
from maxray.inators.base import BaseInator
from maxray.runner import ExecInfo, RunCompleted, RunErrored, RunAborted

import builtins
import io
import os

from pathlib import Path
from functools import partial
from contextlib import contextmanager
from uuid import uuid4
from typing import Iterator, Optional

import click
from loguru import logger
import rerun as rr


class Display(BaseInator):
    def __init__(
        self,
        recording_id: str,
        connect_viewer: Optional[str] = None,
        stream_to_file: Optional[str] = None,
        refresh_interval_ms: float = 500,
    ):
        super().__init__()

        self.recording_id = recording_id
        self.connect_viewer = connect_viewer
        self.stream_to_file = stream_to_file

        self.refresh_interval_ms = refresh_interval_ms

    @staticmethod
    @click.command()
    @click.option("--id", type=str, default=str(uuid4()))
    @click.option("--connect", type=str)
    @click.option("--save", type=str)
    @click.option(
        "--interval",
        type=float,
        default=500,
        help="Display refresh interval (milliseconds)",
    )
    def cli(id: str, connect: Optional[str], save: Optional[str], interval: float):
        # TODO: support connect/save modes
        return Display(
            recording_id=id,
            connect_viewer=connect,
            stream_to_file=save,
            refresh_interval_ms=interval,
        )

    def maxray(self, x, ray: Ray):
        """
        (m)utable xray: this function must return a value to replace `x` in the evaluation of the source expression.

        Coarse control flow can be achieved via `exit()` or raising one of `Break()`, `AbortRun()`, or `RestartRun()`.
        """
        if x is builtins.print:
            x = partial(self.print, ctx=ray.ctx)

        return x

    def xray(self, x, ray: Ray):
        self.last_ctx = ray.ctx

        match x:
            case RunCompleted():
                R.DisplayChannel.push(SetStatus("Completed", "[green]"))
            case RunAborted(exception=exception):
                R.DisplayChannel.push(
                    SetStatus(f"Aborted ({type(exception).__name__})", "[cyan]")
                )
            case RunErrored(exception=exception, traceback=traceback):
                R.DisplayChannel.push(SetStatus("Errored", "[red]"))
                R.DisplayChannel.push(DumpTraceback(exception, traceback))
            case _:
                R.DisplayChannel.push(SetStatus("Running...", "[yellow]"))

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

        rerun_log_level = msg.record["level"].name
        if rerun_log_level == "WARNING":
            rerun_log_level = "WARN"

        rr.log(
            prefix,
            rr.TextLog(
                f"{msg.record['message']}",
                level=rerun_log_level,
            ),
        )

    def print(self, *args, ctx, **kwargs):
        if "file" in kwargs:
            return print(*args, **kwargs)

        print(*args, **kwargs, file=(buf := io.StringIO()))

        logger.bind(
            maxray_logged_from="print",
            source_file=ctx.fn_context.source_file,
            source_line=ctx.location[0],
        ).log("INFO", buf.getvalue().strip())

    @contextmanager
    def enter_session(self, xi: ExecInfo):
        if xi.source_origin is None:
            run_name = "<temporary>"
        else:
            script = xi.source_origin
            while script.name in ["__init__.py", "__main__.py", "src"]:
                script = script.parent
            run_name = script.name

        rr.init(f"xpy:{run_name}", recording_id=self.recording_id)
        if self.stream_to_file is not None:
            rr.save(self.stream_to_file)
        elif self.connect_viewer is not None:
            rr.connect_tcp(self.connect_viewer)
        else:
            rr.spawn(memory_limit="50%")

        assert rr.get_recording_id() is not None

        enable_internal_logging = bool(os.environ.get("MAXRAY_LOG_LEVEL"))
        # configure Rerun as the logging sink instead of stderr
        logger.enable("maxray")
        logger.remove()
        logger.add(
            self.log_message,
            filter=lambda record: "maxray_internal" not in record["extra"]
            or enable_internal_logging,
        )

        try:
            with (
                R.DisplayChannel.stack(
                    partial(self.iter_display_messages, script_name=run_name)
                ),
                super().enter_session(xi),
            ):
                try:
                    yield
                finally:
                    R.DisplayChannel.push(SetStatus("Exited"))
        finally:
            # TODO: reset logger on display exit?
            pass

    def iter_display_messages(
        self,
        messages: Iterator[SetStatus | ShowValue | DumpTraceback],
        *,
        script_name="???.py",
    ):
        current_status_message = "NOT YET STARTED"
        _last_rendered_content = ""

        def render(*components):
            # Only re-log if display contents actually changed
            nonlocal _last_rendered_content

            components_rendered = []
            for c in components:
                match c:
                    # TODO: archetype views
                    # - dataframes -> markdown table
                    # - arrays -> tensors

                    case _:
                        components_rendered.append(repr(c))

            components_content = "\n\n".join(map(str, components))

            text_content = f"""# {script_name}: {current_status_message}

{components_content}"""

            if text_content != _last_rendered_content:
                rr.log(
                    "display/status",
                    rr.TextDocument(text_content, media_type=rr.MediaType.MARKDOWN),
                )
                _last_rendered_content = text_content

        for msg in messages:
            match msg:
                case SetStatus(text=text) if text != current_status_message:
                    current_status_message = text
                    render()

                case ShowValue(value=value):
                    render(repr(value))

                case DumpTraceback(exception=exception, traceback=traceback):
                    # TODO: proper traceback repr
                    render(
                        "## " + repr(exception),
                    )

            yield
