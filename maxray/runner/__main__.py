from . import (
    ScriptFromModule,
    ScriptFromFile,
    ScriptRunner,
    InteractiveRunner,
    WatchHook,
    InstanceWatchHook,
    RunCompleted,
    RunErrored,
    RunAborted,
)
import pyarrow.feather as ft

import sys
from pathlib import Path
from typing import Optional

import click


_DEFAULT_CAPTURE_LOGS_NAME = ".maxray-logs.arrow"


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_interspersed_args=True)
)
@click.argument("script", type=str)
@click.option("-w", "--watch", type=str, multiple=True)
@click.option("-W", "--watch-instance", type=str, multiple=True)
@click.option("-m", "--module", is_flag=True)
@click.option(
    "-c",
    "--capture-default",
    is_flag=True,
    help=f"Save the recorded logs to the same folder as the script with the default name [{_DEFAULT_CAPTURE_LOGS_NAME}]",
)
@click.option("-C", "--capture", type=str, help="Save the recorded logs to this path")
@click.option(
    "-l",
    "--loop",
    is_flag=True,
    help="Won't exit on completion and will wait for a file change event to run the script again.",
)
@click.option(
    "--restrict",
    is_flag=True,
    help="Don't recursively patch source code, only tracing code in the immediately invoked script file",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def cli(
    script: str,
    module: bool,
    watch: tuple,
    watch_instance: tuple,
    loop: bool,
    capture_default: bool,
    capture: Optional[str],
    restrict: bool,
    args,
):
    # Reset sys.argv so client scripts don't try to parse our arguments
    sys.argv = [sys.argv[0], *args]

    if module:
        run = ScriptFromModule(script)
    else:
        run = ScriptFromFile(script)

    hooks = []
    if watch:
        hooks.extend(WatchHook.build(spec) for spec in watch)
    if watch_instance:
        for spec in watch_instance:
            if spec == "-":
                overlay_script = (
                    (script_path := Path(script))
                    .resolve(True)
                    .with_name(f"over_{script_path.name}")
                )
                hooks.append(InstanceWatchHook.build(f"{overlay_script}:Inator"))
            else:
                hooks.append(InstanceWatchHook.build(spec))

    run_wrapper = ScriptRunner(
        run,
        enable_capture=(capture is not None) or capture_default,
        restrict_to_source_module=restrict,
    )

    as_interactive = len(hooks) > 0 or loop
    if as_interactive:
        run_wrapper = InteractiveRunner(run_wrapper, hooks, loop=loop)

    run_result = run_wrapper.run()

    if isinstance(run_result, RunAborted):
        print(run_result.reason)
        # Nothing to save, immediately exit
        if isinstance(run_result.exception, KeyboardInterrupt):
            sys.exit(0)
        else:
            sys.exit(2)

    if capture is not None:
        capture_to = Path(capture)
        ft.write_feather(run_result.logs_arrow, str(capture_to))
        ft.write_feather(
            run_result.functions_arrow,
            str(capture_to.with_stem(capture_to.stem + "-functions")),
        )
    elif capture_default:
        capture_to = (
            Path(run.sourcemap_to()).resolve(True).parent / _DEFAULT_CAPTURE_LOGS_NAME
        )
        ft.write_feather(
            run_result.logs_arrow,
            str(capture_to),
        )
        ft.write_feather(
            run_result.functions_arrow,
            str(capture_to.with_stem(capture_to.stem + "-functions")),
        )

    match run_result:
        case RunErrored():
            import rich
            from rich.traceback import Traceback

            rich.print(Traceback(run_result.exception_trace))
            sys.exit(1)

        case RunCompleted():
            print("RunCompleted()")


def main():
    cli.main(standalone_mode=False)


if __name__ == "__main__":
    main()
