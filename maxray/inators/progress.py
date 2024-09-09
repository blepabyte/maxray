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

import time
from contextlib import contextmanager
from pathlib import Path

import click


class Show(BaseInator):
    def __init__(self, elapsed_show_threshold: float):
        super().__init__()
        self.elapsed_show_threshold = elapsed_show_threshold
        #

    @staticmethod
    @click.command()
    @click.option(
        "-t",
        "--threshold",
        type=float,
        default=1.0,
        help="Show a progress bar if the task is expected to take longer than x seconds",
    )
    def cli(threshold: float):
        return Show(elapsed_show_threshold=threshold)

    def wrapped_iterator(self, it, label: str):
        from rich.progress import Progress

        try:
            n = len(it)
        except Exception:
            n = None

        progress = Progress(auto_refresh=False)
        t1 = progress.add_task(label, total=n)

        st = time.perf_counter()
        count = 0
        bar_shown = False
        try:
            for x in it:
                yield x
                count += 1
                progress.update(t1, advance=1, field_uhh=f"n = {count}")

                n_expected = n if n is not None else 10
                elapsed = time.perf_counter() - st
                if (
                    elapsed > 1
                    or elapsed / count * n_expected > self.elapsed_show_threshold
                ) and not bar_shown:
                    S.display.elements.insert(0, progress)
                    bar_shown = True

        finally:
            try:
                S.display.elements.remove(progress)
            except Exception:
                ...

    def maxray(self, x, ray: Ray):
        ctx = ray.ctx

        match ray.iterated():
            case [it]:
                source_file = Path(ctx.fn_context.source_file).name
                display_label = (
                    f"[yellow]{source_file}:{ctx.location[0]}) {it} in {ctx.source}"
                )
                return self.wrapped_iterator(x, label=display_label)
            case _:
                ...

        return x
