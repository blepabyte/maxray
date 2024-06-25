from maxray import maxray, NodeContext
from maxray.capture.logs import CaptureLogs
from maxray.function_store import FunctionStore

import pyarrow.compute as pc
import pyarrow.feather as ft
import watchfiles

import threading
import importlib
import importlib.util
from pydoc import importfile
import ast
from textwrap import indent
import sys
import tempfile
from contextvars import ContextVar
from typing import Any, Callable, Optional
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from rich.console import Console
import click

from maxray.transforms import FnContext

console = Console()


class EndOfScript: ...


def dump_on_exception():
    console.print_exception(show_locals=True, suppress=["maxray"])


def empty_module(module_name: str):
    module_code = ""

    # Setting module path doesn't make a difference to imports. Need to set sys.path
    use_file_for_module = str(__file__)

    module_ast = ast.parse(module_code, filename=use_file_for_module)
    module_code_object = compile(module_ast, filename=use_file_for_module, mode="exec")

    module = ModuleType(module_name)
    exec(module_code_object, module.__dict__)

    return module


def set_event():
    event = threading.Event()
    event.set()
    return event


@dataclass
class WatchHook:
    watch_file: str
    watch_symbol: str
    last_inator: Callable = lambda x, ctx: x
    needs_reload: threading.Event = field(default_factory=set_event)

    def block_call_until_updated(self, x, ctx: NodeContext):
        self.needs_reload.clear()
        self.needs_reload.wait()
        return self.call_latest(x, ctx)

    def call_latest(self, x, ctx: NodeContext):
        # Lazily import only when actually called
        if self.needs_reload.is_set():
            from_module = importfile(self.watch_file)
            self.last_inator = getattr(from_module, self.watch_symbol)
            self.needs_reload.clear()
        return self.last_inator(x, ctx)

    def check_and_update(self, changed_files):
        if not self.needs_reload.is_set() and self.watch_file in changed_files:
            self.needs_reload.set()

    @staticmethod
    def build(watch_spec: str):
        """
        Watch spec of `foo.py:some_fn` means `some_fn` will be imported from `foo.py`, and reloaded every time that file is changed
        """
        dyn_code_file, watcher_symbol = watch_spec.rsplit(":", maxsplit=1)
        dyn_code_path = Path(dyn_code_file).resolve(True)
        assert dyn_code_path.suffix == ".py"
        return WatchHook(str(dyn_code_path), watcher_symbol)


class ScriptFromFile:
    def __init__(self, script_path):
        self.script_path = Path(script_path).resolve(True)

    def source(self):
        return self.script_path.read_text()

    def sourcemap_to(self):
        return str(self.script_path)

    def extra_sys_path(self):
        return [str(self.script_path.parent)]

    def in_module(self):
        return empty_module("_empty_module")


class ScriptFromModule:
    def __init__(self, module_name: str):
        main_module_name = f"{module_name}.__main__"
        if module_name in sys.modules:
            sys.modules.pop(module_name)
        if main_module_name in sys.modules:
            sys.modules.pop(main_module_name)

        module_spec = importlib.util.find_spec(main_module_name)
        assert (
            module_spec is not None and module_spec.origin is not None
        ), f"Couldn't find a __main__.py for {module_name}"
        self.module_name = module_name
        self.module = importlib.import_module(module_name)
        self.main_path = Path(module_spec.origin).resolve(True)
        assert self.module_name in sys.modules

    def source(self):
        return self.main_path.read_text()

    def sourcemap_to(self):
        return str(self.main_path)

    def extra_sys_path(self):
        return []

    def in_module(self):
        return self.module


class ScriptFromString:
    def __init__(self, python_code: str):
        self.python_code = python_code

        self.script_dir = Path(tempfile.mkdtemp())
        with open(self.script_dir / "__main__.py", "w") as f:
            f.write(self.python_code)

    def source(self):
        return self.python_code

    def sourcemap_to(self):
        return None

    def extra_sys_path(self):
        return []

    def in_module(self):
        return empty_module("_empty_module")


@dataclass
class ScriptOutput:
    logs_arrow: Any
    functions_arrow: Any
    exception: Any = None
    final_scope: dict = field(default_factory=dict)

    def completed(self):
        return self.exception is None


@dataclass
class ScriptRunner:
    """
    Supports programmatically running Python scripts and modules in a __main__ context with specified maxray transforms applied.
    """

    MAIN_FN_NAME = "maaaaaaaaaaaaaaaaaaaaaaaaaaaain"

    run_type: ScriptFromFile | ScriptFromString | ScriptFromModule
    with_decor_inator: Callable = lambda x, ctx: x
    temp_sourcefile: Any = field(
        default_factory=lambda: tempfile.NamedTemporaryFile("w", delete=False)
    )

    @staticmethod
    def run_script(script_path, **kwargs):
        return ScriptRunner(ScriptFromFile(script_path), **kwargs).run()

    @staticmethod
    def run_module(module_name):
        return ScriptRunner(ScriptFromModule(module_name)).run()

    @staticmethod
    def run_code(python_code):
        return ScriptRunner(ScriptFromString(python_code)).run()

    def compile_to_callable(self):
        source = self.run_type.source()
        new_source = f"""def {self.MAIN_FN_NAME}():
{indent(source, '    ')}
    return locals()
"""

        # inspect.getsource relies on the file actually existing
        with open(self.temp_sourcefile.name, "w") as f:
            f.write(new_source)

        tree = ast.parse(new_source, filename=self.temp_sourcefile.name)
        main_func = ast.fix_missing_locations(tree)

        # BUG: will error if `import *` is used (doesn't work inside a fn `def`)
        compiled_code = compile(
            main_func,
            filename=self.temp_sourcefile.name,
            mode="exec",
        )

        namespace = {}
        exec(compiled_code, namespace)
        main = namespace[self.MAIN_FN_NAME]

        # internals require module to be valid for transform to go ahead
        exec_in_module = self.run_type.in_module()
        sys.modules[exec_in_module.__name__] = exec_in_module
        main.__module__ = exec_in_module.__name__
        return main

    def run(self):
        # Fixes same-directory imports (as the actual script is in some random temp dir)
        prev_sys_path = [*sys.path]
        sys.path.extend(self.run_type.extra_sys_path())

        exc = None
        final_scope = {}
        try:
            fn = maxray(self.rewrite_node, initial_scope={"__name__": "__main__"})(
                self.compile_to_callable()
            )

            with CaptureLogs() as cl:
                try:
                    final_scope = fn()

                    # A way to run custom code at the end of a script
                    self.with_decor_inator(
                        EndOfScript(),
                        NodeContext(
                            "internal/capture",
                            "EndOfScript()",
                            FnContext(
                                EndOfScript.__init__,
                                "__init__",
                                "maxray.capture",
                                "",
                                "",
                                ContextVar("maxray_call_counter", default=0),
                            ),
                            (0, 0, 0, 0),
                        ),
                    )
                except Exception as e:
                    dump_on_exception()
                    exc = e
                    final_scope = {}

        finally:
            sys.path = prev_sys_path

        functions_table = FunctionStore.collect()
        # Patch the correct source file names (temporary -> actual)
        sf_col_idx = functions_table.column_names.index("source_file")
        remapped_source_file = pc.replace_substring_regex(
            functions_table["source_file"],
            pattern=rf"^{self.temp_sourcefile.name}$",
            replacement=f"{self.run_type.sourcemap_to()}",
        )
        functions_table = functions_table.set_column(
            sf_col_idx, functions_table.field(sf_col_idx), remapped_source_file
        )

        return ScriptOutput(cl.collect(), functions_table, exc, final_scope)

    def rewrite_node(self, x, ctx: NodeContext):
        # functions and contextvars can't be deepcopied
        ctx = copy(ctx)
        ctx.fn_context = copy(ctx.fn_context)

        if ctx.fn_context.source_file == self.temp_sourcefile.name:
            ctx.fn_context.source_file = self.run_type.sourcemap_to()

            # subtract the "def" line and new indentation
            ctx.location = (
                ctx.location[0] - 1,
                ctx.location[1] - 1,
                ctx.location[2] - 4,
                ctx.location[3] - 4,
            )

        CaptureLogs.extractor(x, ctx)

        x = self.with_decor_inator(x, ctx)

        return x


class InteractiveRunner:
    """
    Implements hot-reloading via file-watching
    """

    def __init__(
        self,
        run_type: ScriptFromFile | ScriptFromString | ScriptFromModule,
        watch_hooks: list[WatchHook],
        loop: bool = False,
    ):
        self.run_type = run_type
        self.runner = ScriptRunner(
            self.run_type, with_decor_inator=self.apply_interactive_inators
        )

        self.watch_hooks = watch_hooks
        self.loop = loop

    def run(self):
        watcher = threading.Thread(target=self.watch_loop, daemon=True)
        # Well, we can't kill a thread so we'll just let it die on program exit...
        # BUG: pthreads stuff...
        # FATAL: exception not rethrown
        # fish: Job 1, 'xpy -w examples/hotreload_obser…' terminated by signal SIGABRT (Abort)
        watcher.start()

        while True:
            # Exceptions are absorbed into the run output
            result = self.runner.run()

            if not self.loop:
                return result

            print("\n" * 3)
            with console.status(
                "Iteration in --loop mode completed: [bold green]Watching for source file changes..."
            ) as _status:
                console.log("Press Ctrl-C to quit")
                self.block_until_update()

    def block_until_update(self):
        """
        Wait until either the script file or one of the watched files is modified.
        """
        files_to_watch = [hook.watch_file for hook in self.watch_hooks]
        if (source_file := self.run_type.sourcemap_to()) is not None:
            files_to_watch.append(source_file)

        for _event in watchfiles.watch(*files_to_watch):
            return

    def watch_loop(self):
        # Support multiple scripts, each of which can be hot-reloaded individually
        files = [Path(hook.watch_file) for hook in self.watch_hooks]

        for change in watchfiles.watch(*files):
            changed_files = set([str(change[1]) for change in change])
            for hook in self.watch_hooks:
                hook.check_and_update(changed_files)

    def apply_interactive_inators(self, x, ctx):
        for hook in self.watch_hooks:
            try:
                x = hook.call_latest(x, ctx)
            except Exception:
                dump_on_exception()

                while True:
                    try:
                        x = hook.block_call_until_updated(x, ctx)
                        break
                    except Exception:
                        dump_on_exception()
        return x


_DEFAULT_CAPTURE_LOGS_NAME = ".maxray-logs.arrow"


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_interspersed_args=True)
)
@click.argument("script", type=str)
@click.option("-w", "--watch", type=str, multiple=True)
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
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def cli(
    script: str,
    module: bool,
    watch: tuple,
    loop: bool,
    capture_default: bool,
    capture: Optional[str],
    args,
):
    # Reset sys.argv so client scripts don't try to parse our arguments
    sys.argv = [sys.argv[0], *args]

    if module:
        run = ScriptFromModule(script)
    else:
        run = ScriptFromFile(script)

    if not watch:
        wrapper = ScriptRunner(run)
    else:
        hooks = [WatchHook.build(spec) for spec in watch]
        wrapper = InteractiveRunner(run, hooks, loop=loop)

    _result = wrapper.run()

    if capture is not None:
        ft.write_feather(_result.logs_arrow, capture)
    elif capture_default:
        ft.write_feather(
            _result.logs_arrow,
            str(
                Path(run.sourcemap_to()).resolve(True).parent
                / _DEFAULT_CAPTURE_LOGS_NAME
            ),
        )


def run_script():
    cli()
