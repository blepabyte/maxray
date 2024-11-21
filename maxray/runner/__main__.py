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

from importlib.util import find_spec
import sys
from pathlib import Path

from loguru import logger
import click


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_interspersed_args=True)
)
@click.argument("script", type=str)
@click.option("-W", "--watch", type=str, multiple=True)
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
    "--local",
    is_flag=True,
    help="Only patch and trace source code in the immediately invoked script or module (an alias for `-i .`)",
)
@click.option(
    "-i",
    "--include",
    type=str,
    multiple=True,
    help="By default all functions are recursively patched. If any include flags are manually passed, only functions belonging to included modules are allowed to be patched. Use `-i .` to only include the script/module being run.",
)
@click.option(
    "-x",
    "--exclude",
    type=str,
    multiple=True,
    help="Forbid modules passed in this list from being patched (in addition to those hard-coded in _GLOBAL_SKIP_MODULES)",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Disables display and interactive elements.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def cli(
    script: str,
    module: bool,
    watch: tuple,
    loop: bool,
    local: bool,
    include: tuple,
    exclude: tuple,
    preserve: bool,
    quiet: bool,
    args,
):
    # Reset sys.argv so client scripts don't try to parse our arguments
    sys.argv = [sys.argv[0], *args]

    if module:
        run = ScriptFromModule(script)
    else:
        run = ScriptFromFile(script)

    has_display = False

    hooks = []
    if watch:
        for spec in watch:
            hooks.append(ReloadHook(reloadable := reloadable_from_spec(spec).unwrap()))
            has_display = has_display or reloadable.name == "Display"

    # Each display backend should enable and redirect logs to their display
    if not quiet and not has_display:
        hooks.insert(
            0,
            ReloadHook(reloadable_from_spec("maxray.inators.rich:Display").unwrap()),
        )

    if hooks and not quiet:
        if find_spec("ipdb") is not None:
            hooks.append(
                ReloadHook(reloadable_from_spec("maxray.inators.debug:IPDB").unwrap())
            )

    if local:
        include = include + (".",)

    run_wrapper = ScriptRunner(
        run,
        include_patch_modules=include,
        exclude_patch_modules=exclude,
        preserve_values=preserve,
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
                max_frames=10,
            )
            rich.print(traceback)
            sys.exit(1)

        case RunCompleted():
            print("RunCompleted()")


def main():
    cli.main(standalone_mode=False)


if __name__ == "__main__":
    main()
