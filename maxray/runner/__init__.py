from contextlib import contextmanager
from .exec_sources import ExecSource, DynamicSymbol
from maxray import maxray, NodeContext, _inator_inator
from maxray.capture.logs import CaptureLogs
from maxray.transforms import FnContext
from maxray.function_store import FunctionStore

import pyarrow.compute as pc
import watchfiles

import itertools
import threading
import importlib
import importlib.util
import ast
from textwrap import indent
import sys
import inspect
import tempfile
import time
from contextvars import ContextVar
from typing import Any, Callable, Optional
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from rich.live import Live
from rich.traceback import Traceback, Trace
from rich.pretty import Pretty
from rich.layout import Layout
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich._inspect import Inspect


MAIN_FN_NAME = "maaaaaaain"


# Pauses execution
class Break(Exception): ...


# Should not be caught by any user code
class AbortRun(BaseException): ...


# Should not be caught by any user code
class RestartRun(BaseException): ...


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


def watch_files(file_paths):
    """
    Watches a set of file paths for changes.

    Needed because `watchfiles` (inotify backend) watches file *inodes*, so the watch is lost when the file gets deleted (e.g. on save and rewrite from most text editors)

    Args:
        file_paths: Any iterable of `PathLike`s to existing files to be watched for changes.

    Yields:
        Exact subsets of `file_paths` receiving at least one change event since the last iteration.

        Change types (added, modified, deleted) are discarded because they're unreliable/non-synchronised anyway.

    Notes:
        Includes debouncing of 50ms.
    """
    canon_file_map = {str(Path(p).resolve(True)): p for p in file_paths}
    containing_dirs = list({str(Path(p).parent.resolve()) for p in file_paths})

    for changes in watchfiles.watch(
        *containing_dirs, recursive=False, step=50, debounce=1000
    ):
        changed_files = (
            # Can't do strict resolve here because it might not exist
            canon_file_map.get(str(Path(c[1]).resolve()), None)
            for c in changes
        )
        yield set(filter(bool, changed_files))


@dataclass
class WatchHook:
    watched: DynamicSymbol
    last_inator: Callable = lambda x, ctx: x
    needs_reload: threading.Event = field(default_factory=set_event)

    def block_call_until_updated(self, x, ctx: NodeContext):
        self.needs_reload.clear()
        self.needs_reload.wait()
        return self.call_latest(x, ctx)

    def require_reload(self):
        self.needs_reload.set()

    def call_latest(self, x, ctx: NodeContext):
        # Lazily import only when actually called
        if self.needs_reload.is_set():
            self.last_inator = self.watched.load()
            self.needs_reload.clear()
        return self.last_inator(x, ctx)

    def check_and_update(self, changed_files):
        if not self.needs_reload.is_set() and self.watched.is_changed_by_files(
            changed_files
        ):
            self.needs_reload.set()

    @staticmethod
    def build(watch_spec: str):
        """
        Watch spec of `foo.py:some_fn` means `some_fn` will be imported from `foo.py`, and reloaded every time that file is changed
        """
        return WatchHook(DynamicSymbol.from_spec(watch_spec, "maxray.inators").unwrap())


@dataclass
class InstanceWatchHook:
    watched: DynamicSymbol
    state_instance: Optional[Callable] = None
    needs_reload: threading.Event = field(default_factory=set_event)

    def block_call_until_updated(self, x, ctx: NodeContext):
        self.needs_reload.clear()
        self.needs_reload.wait()
        return self.call_latest(x, ctx)

    def require_reload(self):
        self.needs_reload.set()
        # self.state_instance = None

    def call_latest(self, x, ctx: NodeContext):
        # Lazily import only when actually called
        if self.state_instance is None or self.needs_reload.is_set():
            state_type = self.watched.load()
            if self.state_instance is None:
                self.state_instance = state_type()
            else:
                # Monkey-patch to preserve state between hot reloads
                self.state_instance.__class__ = state_type
            self.needs_reload.clear()
        return self.state_instance(x, ctx)

    def check_and_update(self, changed_files):
        if not self.needs_reload.is_set() and self.watched.is_changed_by_files(
            changed_files
        ):
            self.needs_reload.set()

    @staticmethod
    def build(watch_spec: str):
        """
        Watch spec of `foo.py:some_fn` means `some_fn` will be imported from `foo.py`, and reloaded every time that file is changed
        """
        return InstanceWatchHook(
            DynamicSymbol.from_spec(watch_spec, "maxray.inators").unwrap()
        )


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
class RunCompleted:
    logs_arrow: Any
    functions_arrow: Any
    final_scope: dict = field(default_factory=dict)


@dataclass
class RunErrored:
    logs_arrow: Any
    functions_arrow: Any
    exception: Exception
    exception_trace: Trace


@dataclass
class RunAborted:
    reason: str
    exception: Optional[BaseException] = None


@dataclass
class ScriptRunner:
    """
    Supports programmatically running Python scripts and modules in a __main__ context with specified maxray transforms applied.
    """

    run_type: ScriptFromFile | ScriptFromString | ScriptFromModule
    enable_capture: bool = True
    with_decor_inator: Callable = lambda x, ctx: x
    restrict_to_source_module: bool = False
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
        # HACK: stupid hack (since everything gets wrapped in a function, globals become nonlocals)
        # TODO: support passing AST mods to transform
        # BUG: won't show up in `globals()`
        source = source.replace("global ", "nonlocal ")

        new_source = f"""def {MAIN_FN_NAME}():
{indent(source, '    ')}
    # return locals()
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

        # internals require module to be valid for transform to go ahead
        exec_in_module = self.run_type.in_module()

        namespace = exec_in_module.__dict__
        exec(compiled_code, namespace)
        main = namespace[MAIN_FN_NAME]

        sys.modules[exec_in_module.__name__] = exec_in_module
        main.__module__ = exec_in_module.__name__
        return main

    def run(self) -> RunCompleted | RunErrored | RunAborted:
        # Fixes same-directory imports (as the actual script is in some random temp dir)
        prev_sys_path = [*sys.path]
        sys.path.extend(self.run_type.extra_sys_path())

        push_main_scope = {
            "__name__": "__main__",
            "__file__": self.run_type.sourcemap_to(),
        }

        # Try to evaluate source (e.g. could fail with SyntaxError)
        try:
            fn = maxray(
                self.rewrite_node,
                pass_scope=True,
                root_inator=_inator_inator(
                    [self.run_type.in_module().__name__]
                    if self.restrict_to_source_module
                    else None
                ),
                initial_scope=push_main_scope,
            )(self.compile_to_callable())
        except KeyboardInterrupt as e:
            return RunAborted("Signal interrupt", e)
        except BaseException as e:
            sys.path = prev_sys_path
            return RunAborted(f"Parse/eval of source {self.run_type} failed", e)

        # HACKY exception logic
        # click gives us no choice because it *insists* on stealing KeyboardInterrupt even in standalone mode and throwing SystemExit
        with CaptureLogs() as cl:
            try:
                final_scope = fn()

            except Exception as e:
                exc_trace = Traceback.extract(
                    type(e), e, e.__traceback__, show_locals=True
                )
                return self.run_end_hook(
                    RunErrored(cl.collect(), self.collect_functions(), e, exc_trace)
                )

            except (AbortRun, RestartRun) as e:
                # this is the only way to quit the run as `exit` will not work
                return RunAborted("User abort", e)

            except KeyboardInterrupt as e:
                return RunAborted("Signal interrupt", e)

            except SystemExit as e:
                return RunAborted("Forced exit", e)

            except BaseException as e:
                return RunAborted("BaseException with unknown intent", e)

        return self.run_end_hook(
            RunCompleted(cl.collect(), self.collect_functions(), final_scope)
        )

    def run_end_hook(self, status: RunCompleted | RunErrored):
        """
        Run custom code at the end of a script
        """
        self.with_decor_inator(
            status,
            NodeContext(
                "maxray/runner",
                f"{type(status).__name__}(...)",
                FnContext(
                    type(status).__init__,
                    "__init__",
                    "maxray.runner",
                    "",
                    "",
                    0,
                    call_count=ContextVar("maxray_call_counter", default=0),
                    compile_id="00000000-0000-0000-0000-000000000000",
                ),
                (0, 0, 0, 0),
            ),
        )
        return status

    def collect_functions(self):
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
        return functions_table

    def rewrite_node(self, x, ctx: NodeContext):
        if ctx.fn_context.source_file == self.temp_sourcefile.name:
            # copy if needed to prevent mutation

            ctx = copy(ctx)
            ctx.fn_context = copy(
                ctx.fn_context
            )  # functions and contextvars can't be deepcopied

            ctx.fn_context.source_file = self.run_type.sourcemap_to()

            # subtract the "def" line and new indentation
            ctx.location = (
                ctx.location[0] - 1,
                ctx.location[1] - 1,
                ctx.location[2] - 4,
                ctx.location[3] - 4,
            )

        if self.enable_capture:
            CaptureLogs.extractor(x, ctx)

        x = self.with_decor_inator(x, ctx)

        return x


class BaseInteractiveContext:
    def display_started(self):
        raise NotImplementedError()

    def display_end_status(self, status: RunCompleted | RunErrored):
        raise NotImplementedError()


class InteractiveContext(BaseInteractiveContext):
    def __init__(self):
        self.live = Live(
            Pretty("Waiting for data to show..."), screen=True, auto_refresh=False
        )
        self.ctx = None
        self.status = "[yellow]Running..."

        self.var_displays = {}
        self.tracked = {}

    def copy(self):
        return self.ctx

    def display_started(self):
        self.status = "[yellow]Running..."
        self.display()

    def display_end_status(self, status: RunCompleted | RunErrored | RunAborted):
        with self.with_display():
            match status:
                case RunCompleted():
                    self.status = "[green]Completed"
                    self.display()
                case RunErrored():
                    self.status = "[red]Errored"
                    self.clear()
                    self.display(exception=status.exception_trace)
                case RunAborted():
                    self.status = f"[cyan]Aborted ({type(status.exception).__name__})"
                    self.clear()
                    self.display()

    def dashboard(self):
        return self.live

    def wrap(self, ctx: NodeContext):
        assert isinstance(ctx, NodeContext)
        self.ctx = ctx
        return self

    def __getattr__(self, prop):
        return getattr(self.ctx, prop)

    def show(self, obj):
        self.var_displays["show"] = Pretty(obj)
        self.display()

    def inspect(self, obj):
        self.var_displays["inspect"] = Inspect(obj, all=False, methods=True, docs=True)
        self.display()

    def track(self, **keys):
        self.tracked.update(keys)
        self.display()

    def clear(self):
        self.var_displays.clear()
        self.tracked.clear()

    def current_code(self):
        if self.ctx is None:
            return Text("No code for current context")

        ctx = self.ctx
        offset_line = (
            ctx.location[0] - ctx.fn_context.line_offset + 3
        )  # TODO: 3 in main script, 2 elsewhere
        return Syntax(
            ctx.fn_context.source,
            "python",
            line_numbers=True,
            line_range=(max(1, offset_line - 5), offset_line + 5),
            highlight_lines={offset_line},
        )

    def current_stack(self):
        # TODO: filter stack properly
        text = "\n".join(f.function for f in inspect.stack()[7:])
        return Text(text)

    def current_tracked(self):
        table = Table(expand=True, title="Tracked objects")
        table.add_column(
            "Id",
            ratio=1,
        )
        table.add_column("Object", ratio=4)

        assume_terminal_height = 128
        row_height = max(5, assume_terminal_height // (len(self.tracked) + 1))
        # dicts are insertion-ordered: reverse to show newest first (overflow is hidden outside terminal bounds)
        for k, v in reversed(self.tracked.items()):
            table.add_row(
                str(k),
                Panel(Pretty(v), height=row_height, padding=(0, 0)),
            )
        return table

    @contextmanager
    def with_display(self):
        try:
            yield
        except Exception as e:
            self.display(exception=True)
            raise e

    def bsod(self):
        # When trying to display the dashboard itself errors
        pass

    def display(self, exception=False):
        root_layout = Layout()
        root_layout.split_column(
            Layout(name="header"), main_layout := Layout(name="content")
        )
        root_layout["header"].size = 15
        header_table = Table(expand=True, title=self.status)

        if self.ctx is None:
            current_source = ""
            current_file = "-"
        else:
            current_source = self.ctx.source
            current_file = "/".join(Path(self.ctx.fn_context.source_file).parts[-3:])

        header_table.add_column(current_file, ratio=2)
        header_table.add_column("Source", ratio=1)
        header_table.add_column("Stack", ratio=2)

        header_table.add_row(
            self.current_code(),
            current_source,
            self.current_stack(),
        )
        root_layout["header"].update(header_table)

        if exception:
            if exception is True:
                traceback = Traceback(suppress=[sys.modules["maxray"]])
            else:
                traceback = Traceback(
                    exception,
                    suppress=[sys.modules["maxray"]],
                    show_locals=True,
                )
            left, right = Layout(), Layout(traceback)
            main_layout.split_row(left, right)
            main_layout = left

        for k, v in self.var_displays.items():
            push_layout = Layout(Panel(v, title=k))
            if isinstance(v, Inspect):
                push_layout.ratio = 4
            main_layout.add_split(push_layout)

        if self.tracked:
            push_layout = Layout(self.current_tracked())
            push_layout.ratio = 4
            main_layout.add_split(push_layout)

        self.live.update(root_layout)
        self.live.refresh()


class InteractiveRunner:
    """
    Implements hot-reloading via file-watching
    """

    def __init__(
        self,
        runner: ScriptRunner,
        watch_hooks: list[WatchHook],
        loop: bool = False,
    ):
        self.runner = runner
        self.runner.with_decor_inator = self.apply_interactive_inators
        self.run_type = self.runner.run_type

        self.watch_hooks = watch_hooks
        self.loop = loop

        self.interactive_state = InteractiveContext()

    def run(self) -> RunCompleted | RunErrored | RunAborted:
        watcher = threading.Thread(target=self.watch_loop, daemon=True)
        # Well, we can't kill a thread so we'll just let it die on program exit...
        # BUG: pthreads stuff...
        # FATAL: exception not rethrown
        # fish: Job 1, 'xpy -w examples/hotreload_obser…' terminated by signal SIGABRT (Abort)
        watcher.start()

        # TODO: swappable interactive backends
        with self.interactive_state.dashboard():
            while True:
                self.interactive_state.display_started()

                for hook in self.watch_hooks:
                    hook.require_reload()

                # All exceptions should be absorbed into the run output
                result = self.runner.run()
                self.interactive_state.display_end_status(result)

                match result:
                    case RunAborted(exception=AbortRun()):
                        if self.loop:
                            self.block_until_update()
                            continue
                        else:
                            return result
                    case RunAborted(exception=RestartRun()):
                        continue
                    case RunAborted():  # general abort
                        return result
                    case _:
                        pass

                if not self.loop:
                    # Delay makes it less confusing when the program immediately exits
                    time.sleep(2)
                    return result

                # with console.status(
                #     "Iteration in --loop mode completed: [bold green]Watching for source file changes..."
                # ) as _status:
                # console.log("Press Ctrl-C to quit")
                self.block_until_update()

    def block_until_update(self):
        """
        Wait until either the script file or one of the watched files is modified.
        """
        # TODO: print a message here? multiple interact backends required?
        files_to_watch = list(
            itertools.chain.from_iterable(
                hook.watched.files_to_watch() for hook in self.watch_hooks
            )
        )
        if (source_file := self.run_type.sourcemap_to()) is not None:
            files_to_watch.append(source_file)

        for _event in watchfiles.watch(*files_to_watch):
            return

    def watch_loop(self):
        # Support multiple scripts, each of which can be hot-reloaded individually
        files = list(
            itertools.chain.from_iterable(
                hook.watched.files_to_watch() for hook in self.watch_hooks
            )
        )

        for changed_files in watch_files(files):
            for hook in self.watch_hooks:
                hook.check_and_update(changed_files)

    def apply_interactive_inators(self, x, ctx):
        ctx = self.interactive_state.wrap(ctx)

        for hook in self.watch_hooks:
            try:
                x = hook.call_latest(x, ctx)
                continue
            except Exception as e:
                ctx.status = "[violet]PAUSED"
                # Traceback is unhelpful for breaks - the PAUSED status update suffices
                ctx.display(exception=not isinstance(e, Break))

            # Don't nest in `except` block to avoid "During handling of the above exception" messing up the traceback display
            while True:
                try:
                    x = hook.block_call_until_updated(x, ctx)
                    ctx.status = "[yellow]Running..."
                    ctx.display()
                    break
                except Exception as e:
                    ctx.display(exception=not isinstance(e, Break))

        return x
