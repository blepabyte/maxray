from maxray.inators.core import S, Ray
from maxray.inators.base import BaseInator
from maxray.runner import (
    MAIN_FN_NAME,
    RunAborted,
    RunCompleted,
    RunErrored,
    AbortRun,
    RestartRun,
    Break,
)

from uuid import uuid4
from contextlib import contextmanager

import rerun as rr


class Inator(BaseInator):
    def xray(self, x, ray: Ray):
        S.define_once(
            "RERUN_INSTANCE",
            lambda _: rr.init(f"{self}", spawn=True, recording_id=str(uuid4())),
            v=1,
        )

    def maxray(self, x, ray: Ray):
        ctx = ray.ctx

        # Manual control flow
        # exit()
        # raise Break()
        # raise AbortRun()
        # raise RestartRun()

        # Global source code overlays
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
                S.display(df)
            case _:
                ...

        return x

    def runner(self):
        # WARNING: Changes to this function are NOT applied on reload
        # You should modify `match_run_result` below instead
        try:
            while True:
                # Each iteration yields another run of the program
                result: RunCompleted | RunAborted | RunErrored = yield
                match result:
                    case RunAborted(exception=RestartRun()):
                        continue
                    case RunAborted(exception=AbortRun()):
                        ...
                    case RunAborted():
                        return  # Unhandleable error

                self.match_run_result(result)

        finally:
            # Cleanup logic either here or in self.enter_session (contextmanager)
            ...

    def match_run_result(self, result: RunCompleted | RunAborted | RunErrored):
        match result:
            case RunCompleted():
                ...
            case RunErrored():
                with S.display.hidden():
                    result.show_traceback()
                    S.enter_debugger(post_mortem=result.traceback)

        self.wait_and_reload()
