from dataclasses import dataclass, field
from pathlib import Path
from maxray import maxray, xray, NodeContext
from maxray.capture.logs import CaptureLogs

import watchfiles

import threading
import importlib
import importlib.util
from pydoc import importfile
import ast
from textwrap import indent
import sys
import tempfile
from typing import Any, Callable, Optional
from copy import copy
from types import ModuleType

from loguru import logger
from rich.console import Console
import click

console = Console()


def dump_on_exception():
    console.print_exception(show_locals=True, suppress=["maxray"])


@dataclass
class WatchHook:
    watch_file: str
    watch_symbol: str
    latest_callable: Callable  # (x, ctx) -> x | else
    needs_reload: threading.Event = field(default_factory=threading.Event)

    def __call__(self, x, ctx: NodeContext):
        # Lazily import only when actually called
        return self.reloaded().latest_callable(x, ctx)

    def reloaded(self):
        if self.needs_reload:
            from_module = importfile(self.watch_file)
            self.latest_callable = getattr(from_module, self.watch_symbol)
            self.needs_reload.clear()
        return self

    def should_update(self, changed_files):
        print(self.watch_file, changed_files, self.needs_reload)
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
        return WatchHook(str(dyn_code_path), watcher_symbol, None).reloaded()


@dataclass
class WrapScriptIntoMain:
    MAIN_FN_NAME = "maaaaaaaaaaaaaaaaaaaaaaaaaaaain"

    """
    Convenience utility: Sometimes you just want to profile/annotate a script once without modifying any source code.
    This makes a copy of the script file, wrapping its contents in a callable "main" function to which `maxray` transforms can then be applied.
    """

    script_path: str
    temp_sourcefile: Any = field(
        default_factory=lambda: tempfile.NamedTemporaryFile("w", delete=False)
    )
    module: Any = None
    watch_hooks: list[WatchHook] = field(default_factory=list)

    @property
    def script_dir(self):
        return Path(self.script_path).resolve(True).parent

    def build(self):
        with open(self.script_path, "r") as file:
            source = file.read()

        new_source = f"""def {self.MAIN_FN_NAME}():
{indent(source, '    ')}
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

        if self.module is None:
            # assign some dummy module that doesn't conflict
            sys.modules["override_mod"] = self.empty_module("override_mod")
            main.__module__ = "override_mod"
        else:
            sys.modules[self.module.__name__] = self.module
            main.__module__ = self.module.__name__
        return main

    def watch_loop(self):
        # Support multiple scripts, each of which can be hot-reloaded individually
        files = [Path(hook.watch_file) for hook in self.watch_hooks]

        for change in watchfiles.watch(*files):
            changed_files = set([str(change[1]) for change in change])
            for hook in self.watch_hooks:
                hook.should_update(changed_files)

    def run(self):
        watcher = threading.Thread(target=self.watch_loop, daemon=True)
        watcher.start()

        # Fixes file imports (as the actual script is in some random temporary dir)
        sys.path.append(str(self.script_dir))
        fn = maxray(self.rewrite_node, initial_scope={"__name__": "__main__"})(
            self.build()
        )

        with CaptureLogs(self.script_path) as cl:
            fn()

        # Well, we can't kill a thread so we'll just let it die on program exit...
        # BUG: pthreads stuff...
        # FATAL: exception not rethrown
        # fish: Job 1, 'xpy -w examples/hotreload_obser…' terminated by signal SIGABRT (Abort)

        # TODO: return some kind of results handle?

    def rewrite_node(self, x, ctx: NodeContext):
        # functions and contextvars can't be deepcopied
        ctx = copy(ctx)
        ctx.fn_context = copy(ctx.fn_context)

        # source_file should never be self.script_path because we've copied
        if ctx.fn_context.source_file == self.temp_sourcefile.name:
            ctx.fn_context.source_file = self.script_path

            # subtract the "def" line and new indentation
            ctx.location = (
                ctx.location[0] - 1,
                ctx.location[1] - 1,
                ctx.location[2] - 4,
                ctx.location[3] - 4,
            )
        CaptureLogs.extractor(x, ctx)

        for hook in self.watch_hooks:
            while True:
                try:
                    x = hook(x, ctx)
                    break
                except Exception as e:
                    dump_on_exception()
                    print(
                        "Hook failed. Waiting for you to modify the code and retry..."
                    )
                    hook.needs_reload.clear()
                    hook.needs_reload.wait()

        return x

    def empty_module(self, module_name: str):
        module_code = ""

        # Setting module path doesn't make a difference to imports. Need to set sys.path
        module_at = (self.script_dir / module_name).with_suffix(".py")
        use_filename = str(module_at.name)

        module_ast = ast.parse(module_code, filename=use_filename)
        module_code_object = compile(module_ast, filename=use_filename, mode="exec")

        module = ModuleType(module_name)

        exec(module_code_object, module.__dict__)

        return module


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_interspersed_args=True)
)
@click.argument("script", type=str)
@click.option("-w", "--watch", type=str, multiple=True)
@click.option("-m", "--module", is_flag=True)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def cli(script: str, module: bool, watch: tuple, args):
    # Reset sys.argv so client scripts don't try to parse our arguments
    sys.argv = [sys.argv[0], *args]

    if module:
        module_name = script
        module_spec = importlib.util.find_spec(f"{module_name}.__main__")
        assert (
            module_spec is not None and module_spec.origin is not None
        ), f"Couldn't find the __main__.py for {module_name}"
        use_module = importlib.import_module(module_name)
        script_path = Path(module_spec.origin)
    else:
        use_module = None
        script_path = Path(script)

    script_path = script_path.resolve(True)
    assert script_path.suffix == ".py"

    if not watch:
        wrapper = WrapScriptIntoMain(str(script_path), module=use_module)
    else:
        # On a change to a file, lazy flag is set to reload relevant hooks
        hooks = [WatchHook.build(spec) for spec in watch]
        wrapper = WrapScriptIntoMain(
            str(script_path), watch_hooks=hooks, module=use_module
        )

    try:
        wrapper.run()
    except Exception:
        dump_on_exception()


def run_script():
    cli()
