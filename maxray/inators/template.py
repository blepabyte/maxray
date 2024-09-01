from maxray.inators.core import S, Matcher
from maxray.runner import (
    MAIN_FN_NAME,
    RunCompleted,
    RunErrored,
    AbortRun,
    RestartRun,
    Break,
)
from maxray.runner import InteractiveContext

import pandas as pd
import numpy as np

import io
import time
from uuid import uuid4
from pathlib import Path
from functools import partial

import rerun as rr


class Inator:
    def __init__(self):
        self.session_name = f"maxray:{type(self).__name__}"
        self.match_assignments = True
        self.last_display_tick = time.perf_counter()

    def log(self, obj, level="INFO"):
        rr.log("log", rr.TextLog(str(obj), level=level))
        return obj

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

    def __call__(self, x, ctx: InteractiveContext):
        S.define_once(
            "RERUN_INSTANCE",
            lambda _: rr.init(self.session_name, spawn=True, recording_id=str(uuid4())),
            v=1,
        )

        if x is print:
            return partial(self.print, ctx=ctx.copy())

        # Randomly ticks as progress indicator
        if (tick := time.perf_counter()) - self.last_display_tick > 0.1:
            ctx.display()
            self.last_display_tick = tick

        match x:
            # Drop into debugger on unhandled error
            case RunErrored():
                import ipdb
                import rich
                from rich.traceback import Traceback

                ctx.live.stop()
                rich.print(Traceback(x.exception_trace))

                ipdb.post_mortem()

                ctx.live.start()

        # Manual control flow

        # exit()
        # raise Break()
        # raise AbortRun()
        # raise RestartRun()

        # ctx.clear()

        # Global source code overlays
        match ctx.source:
            case "...":
                ...

        # Bind local variables in stack frames we're interested in
        match ctx.local_scope:
            case {} if ctx.fn_context.name == MAIN_FN_NAME:
                ...
            case _:
                return x

        if self.match_assignments:
            # Parse variable assignments
            M = Matcher(x, ctx)
            match M.assigned():
                # case {"df": df}:
                #     ctx.track(df=df)
                case _:
                    ...
            return M.unpacked()

        return x
