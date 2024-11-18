from maxray.inators.core import S, NodeContext, Ray
from maxray.inators.base import BaseInator
from maxray.runner import ExecInfo, RunCompleted, RunErrored, RunAborted

import builtins
import io
import os

from pathlib import Path
from functools import partial
from contextlib import contextmanager
from uuid import uuid4

import click
from loguru import logger
import rerun as rr


class Display(BaseInator):
    def __init__(self, recording_id: str):
        super().__init__()

        self.recording_id = recording_id

    @staticmethod
    @click.command()
    @click.option("--id", type=str, default=str(uuid4()))
    def cli(id: str):
        # TODO: support connect/save modes
        return Display(recording_id=id)

    def maxray(self, x, ray: Ray):
        """
        (m)utable xray: this function must return a value to replace `x` in the evaluation of the source expression.

        Coarse control flow can be achieved via `exit()` or raising one of `Break()`, `AbortRun()`, or `RestartRun()`.
        """
        if x is builtins.print:
            x = partial(self.print, ctx=ray.ctx)

        return x

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
        rr.spawn(memory_limit="50%")

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
            with super().enter_session(xi):
                yield
        finally:
            # TODO: reset logger on display exit?
            pass
