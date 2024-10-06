from maxray.inators.core import S, NodeContext, Ray
from maxray.inators.base import BaseInator

import builtins
import io
import os

from pathlib import Path
from functools import partial

from loguru import logger
import rerun as rr


class Setup(BaseInator):
    def __init__(self):
        super().__init__()

        def log_message(msg):
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

        enable_internal_logging = bool(os.environ.get("MAXRAY_LOG_LEVEL"))
        # configure Rerun as the logging sink instead of stderr
        logger.enable("maxray")
        logger.remove()
        logger.add(
            log_message,
            filter=lambda record: "maxray_internal" not in record["extra"]
            or enable_internal_logging,
        )

    def maxray(self, x, ray: Ray):
        """
        (m)utable xray: this function must return a value to replace `x` in the evaluation of the source expression.

        Coarse control flow can be achieved via `exit()` or raising one of `Break()`, `AbortRun()`, or `RestartRun()`.
        """
        if x is builtins.print:
            x = partial(self.print, ctx=ray.ctx)

        return x

    def print(self, *args, ctx, **kwargs):
        if "file" in kwargs:
            return print(*args, **kwargs)

        print(*args, **kwargs, file=(buf := io.StringIO()))

        logger.bind(
            maxray_logged_from="print",
            source_file=ctx.fn_context.source_file,
            source_line=ctx.location[0],
        ).log("INFO", buf.getvalue().strip())
