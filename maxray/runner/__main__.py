from maxray.runner.exec_sources import reloadable_from_spec
from . import (
    ScriptFromModule,
    ScriptFromFile,
    ScriptRunner,
    InteractiveRunner,
    ReloadHook,
    RunCompleted,
    RunErrored,
    RunAborted,
)
import pyarrow.feather as ft

import sys
from pathlib import Path

import click


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_interspersed_args=True)
)
@click.argument("script", type=str)
@click.option("-W", "--watch", type=str, multiple=True)
# TODO: -x, --exclude-module
@click.option("-m", "--module", is_flag=True)
@click.option(
    "-l",
    "--loop",
    is_flag=True,
    help="Won't exit on completion and will wait for a file change event to run the script again.",
)
@click.option(
    "-p",
    "--preserve",
    is_flag=True,
    help="Don't apply value transformations (e.g. automatically unpacking assignments to allow matching)",
)
@click.option(
    "--restrict",
    is_flag=True,
    help="Don't recursively patch source code, only tracing code in the immediately invoked script file",
)
@click.option(
    "--rerun",
    is_flag=True,
    help="Start a rerun.io viewer (requires `rerun-sdk` to be installed)",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def cli(
    script: str,
    module: bool,
    watch: tuple,
    loop: bool,
    restrict: bool,
    preserve: bool,
    rerun: bool,
    args,
):
    # Reset sys.argv so client scripts don't try to parse our arguments
    sys.argv = [sys.argv[0], *args]

    if module:
        run = ScriptFromModule(script)
        run_name = script
    else:
        run = ScriptFromFile(script)
        run_name = Path(script).name

    hooks = []
    if watch:
        for spec in watch:
            hooks.append(ReloadHook(reloadable_from_spec(spec).unwrap()))

        if hooks and rerun:
            hooks.append(
                ReloadHook(reloadable_from_spec("maxray.inators.rerun:Setup").unwrap())
            )

    run_wrapper = ScriptRunner(
        run,
        restrict_to_source_module=restrict,
        preserve_values=preserve,
    )

    as_interactive = len(hooks) > 0 or loop
    if as_interactive:
        run_wrapper = InteractiveRunner(run_wrapper, hooks, loop=loop)

    if rerun:
        import rerun as rr

        rr.init(f"xpy:{run_name}", spawn=True)

    run_result = run_wrapper.run()

    if isinstance(run_result, RunAborted):
        print(run_result.reason)
        # Nothing to save, immediately exit
        if isinstance(run_result.exception, KeyboardInterrupt):
            sys.exit(0)
        else:
            sys.exit(2)

    match run_result:
        case RunErrored():
            import rich
            from rich.traceback import Traceback

            exc_trace = Traceback.extract(
                type(run_result.exception),
                run_result.exception,
                run_result.traceback,
                show_locals=True,
            )
            traceback = Traceback(
                exc_trace,
                suppress=[sys.modules["maxray"]],
                show_locals=True,
                max_frames=5,
            )
            rich.print(traceback)
            sys.exit(1)

        case RunCompleted():
            print("RunCompleted()")


def main():
    cli.main(standalone_mode=False)


if __name__ == "__main__":
    main()
