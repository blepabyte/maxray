from .exec_sources import (
    ReloadableFunction,
    ReloadableInstance,
    reloadable_from_spec,
)
from maxray import maxray
from maxray.nodes import NodeContext, FnContext, RayContext
from maxray.function_store import FunctionStore
from maxray.inators.core import Ray

import pyarrow.compute as pc
import watchfiles

import itertools
import threading
import importlib
import importlib.util
import ast
from textwrap import indent
import sys
import time
import tempfile
from contextlib import ExitStack
from contextvars import ContextVar
from typing import Any, Callable, Optional
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, TracebackType


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
class ReloadHook:
    watched: ReloadableFunction | ReloadableInstance
    needs_reload: threading.Event = field(default_factory=set_event)

    def wait_for_update(self):
        self.needs_reload.clear()
        self.needs_reload.wait()

    def reload(self):
        while self.watched.reload().is_err():
            # TODO: some indicator that we're waiting?
            self.wait_for_update()

        if isinstance(self.watched, ReloadableInstance):  # patch
            self.watched.instance.wait_and_reload = self.wait_and_reload

        self.needs_reload.clear()

    def wait_and_reload(self):
        self.wait_for_update()
        self.reload()

    def require_reload(self):
        self.needs_reload.set()

    def call_latest(self, x, ray: Ray):
        if self.needs_reload.is_set():
            self.reload()

        return self.watched.call(x, ray)

    def check_and_update(self, changed_files):
        if not self.needs_reload.is_set() and self.watched.triggers_reload(
            changed_files
        ):
            self.needs_reload.set()


class ScriptFromFile:
    def __init__(self, script_path):
        self.script_path = Path(script_path).resolve(True)

    def source(self):
        return self.script_path.read_text()

    def sourcemap_to(self) -> Path:
        return self.script_path

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

    def sourcemap_to(self) -> Path:
        return self.main_path

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
class ExecInfo:
    source_origin: Optional[Path]
    temp_exec_file: Path


@dataclass
class RunCompleted:
    final_scope: dict = field(default_factory=dict)


@dataclass
class RunErrored:
    exception: Exception
    traceback: TracebackType

    def show_traceback(self):
        import rich
        from rich.traceback import Traceback

        exc_trace = Traceback.extract(
            type(self.exception),
            self.exception,
            self.traceback,
            show_locals=True,
        )
        traceback = Traceback(
            exc_trace,
            suppress=[sys.modules["maxray"]],
            show_locals=True,
            max_frames=5,
        )
        rich.print(traceback)


@dataclass
class RunAborted:
    reason: str
    exception: Optional[BaseException] = None


@dataclass
class ScriptRunner:
    """
    Supports programmatically running Python scripts and modules in a __main__ context with specified maxray transforms applied.
    """

    # TODO: remove default args and document
    run_type: ScriptFromFile | ScriptFromString | ScriptFromModule
    with_decor_inator: Callable = lambda x, ray: x
    restrict_to_source_module: bool = False
    temp_sourcefile: Any = field(
        default_factory=lambda: tempfile.NamedTemporaryFile("w", delete=False)
    )
    preserve_values: bool = False

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
            "__file__": str(self.run_type.sourcemap_to()),
        }

        # Try to evaluate source (e.g. could fail with SyntaxError)
        try:
            fn = maxray(
                self.rewrite_node,
                pass_scope=True,
                restrict_modules=(
                    [self.run_type.in_module().__name__]
                    if self.restrict_to_source_module
                    else None
                ),
                initial_scope=push_main_scope,
                preserve_values=self.preserve_values,
            )(self.compile_to_callable())
        except KeyboardInterrupt as e:
            return RunAborted("Signal interrupt", e)
        except BaseException as e:
            sys.path = prev_sys_path
            return RunAborted(f"Parse/eval of source {self.run_type} failed", e)

        # HACKY exception logic
        # click gives us no choice because it *insists* on stealing KeyboardInterrupt even in standalone mode and throwing SystemExit
        try:
            final_scope = fn()

        except Exception as e:
            return self.run_end_hook(
                RunErrored(
                    e,
                    e.__traceback__,
                )
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

        return self.run_end_hook(RunCompleted(final_scope))

    def run_end_hook(self, status: RunCompleted | RunErrored):
        """
        Run custom code at the end of a script
        """
        self.with_decor_inator(
            status,
            RayContext(
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
                        compile_id="00000000-0000-0000-0000-000000000000",
                    ),
                    (0, 0, 0, 0),
                ),
            ),
        )
        return status

    def rewrite_node(self, x, ray: RayContext):
        ctx = ray.ctx
        if ctx.fn_context.source_file == self.temp_sourcefile.name:
            # copy if needed to prevent mutation

            ctx = copy(ctx)
            ctx.fn_context = copy(
                ctx.fn_context
            )  # functions and contextvars can't be deepcopied

            ctx.fn_context.source_file = str(self.run_type.sourcemap_to())

            # subtract the "def" line and new indentation
            ctx.location = (
                ctx.location[0] - 1,
                ctx.location[1] - 1,
                ctx.location[2] - 4,
                ctx.location[3] - 4,
            )

        ray.ctx = ctx
        x = self.with_decor_inator(x, ray)

        return x


class InteractiveRunner:
    """
    Implements hot-reloading via file-watching
    """

    def __init__(
        self,
        runner: ScriptRunner,
        watch_hooks: list[ReloadHook],
        *,
        loop: bool,
    ):
        self.runner = runner
        self.runner.with_decor_inator = self.apply_interactive_inators
        self.run_type = self.runner.run_type

        self.watch_hooks = watch_hooks
        self.loop = loop

    def default_runner(self):
        while True:
            result = yield
            time.sleep(
                1
            )  # Briefly show display to avoid confusion over wtf is going on

            match result:
                case RunAborted(exception=RestartRun()):
                    continue
                case RunAborted(exception=AbortRun()):
                    ...
                case RunAborted():  # general abort
                    return result

            if not self.loop:
                return result

            self.block_until_update()

    def run(self) -> RunCompleted | RunErrored | RunAborted:
        exec_info = ExecInfo(
            self.run_type.sourcemap_to(), Path(self.runner.temp_sourcefile.name)
        )
        watcher = threading.Thread(target=self.watch_loop, daemon=True)
        # Well, we can't kill a thread so we'll just let it die on program exit...
        # BUG: pthreads stuff...
        # FATAL: exception not rethrown
        # fish: Job 1, 'xpy -w examples/hotreload_obser…' terminated by signal SIGABRT (Abort)
        watcher.start()

        run_generators = []
        session_stack = ExitStack()
        for wh in self.watch_hooks:
            if isinstance(wh.watched, ReloadableInstance):
                inator = wh.watched.instance
                session_stack.enter_context(inator.enter_session(exec_info))
                try:
                    run_generators.append(inator.runner())
                except NotImplementedError:
                    continue

        if not run_generators:
            run_generators = [self.default_runner()]

        with session_stack:
            while run_generators:
                run_generator = run_generators.pop()
                try:
                    next(run_generator)
                except StopIteration:
                    continue

                while True:
                    for hook in self.watch_hooks:
                        hook.require_reload()

                    # All exceptions should be absorbed into the run output
                    result = self.runner.run()

                    try:
                        run_generator.send(result)
                    except StopIteration:
                        return result

        raise RuntimeError("No runs were yielded")

    def block_until_update(self):
        """
        Wait until either the script file or one of the watched files is modified.
        """
        # TODO: support watching dirs
        files_to_watch = list(
            itertools.chain.from_iterable(
                hook.watched.depends_on_files for hook in self.watch_hooks
            )
        )
        if (source_file := self.run_type.sourcemap_to()) is not None:
            files_to_watch.append(source_file)

        for _event in watchfiles.watch(*files_to_watch):
            return

    def watch_loop(self):
        # TODO: support watching dirs
        files = list(
            itertools.chain.from_iterable(
                hook.watched.depends_on_files for hook in self.watch_hooks
            )
        )

        for changed_files in watch_files(files):
            for hook in self.watch_hooks:
                hook.check_and_update(changed_files)

    def apply_interactive_inators(self, x, ray: Ray):
        ray.__class__ = Ray

        for hook in self.watch_hooks:
            x = hook.call_latest(x, ray)

        return x
