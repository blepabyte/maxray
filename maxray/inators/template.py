from maxray.inators.core import S
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
from maxray import NodeContext

from uuid import uuid4
from contextlib import contextmanager

import rerun as rr


class Inator(BaseInator):
    def __init__(self):
        super().__init__(
            name="Inator", rerun=True, auto_debug=True, match_assignments=True
        )

    def xray(self, x, ctx: NodeContext):
        S.define_once(
            "RERUN_INSTANCE",
            lambda _: rr.init(f"{self}", spawn=True, recording_id=str(uuid4())),
            v=1,
        )

    def maxray(self, x, ctx: NodeContext):
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

        match ctx.local_scope:
            case {} if ctx.fn_context.name == MAIN_FN_NAME:
                ...
            case _:
                return x

        match self.assigned():
            case {"df": df}:
                self.display(df)
            case _:
                ...

        return x

    def runner(self):
        # WARNING: Changes to this function are NOT applied on reload
        # You should modify `match_run_result` below instead
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

        # Cleanup logic either here or in self.enter_session (contextmanager)
        ...

    def match_run_result(self, result: RunCompleted | RunAborted | RunErrored):
        match result:
            case RunCompleted():
                ...
            case RunErrored():
                self.enter_debugger(post_mortem=result)

        self.wait_and_reload()
