from dataclasses import dataclass, field
from pathlib import Path
from maxray import xray, _set_logging, NodeContext
from maxray.capture.logs import CaptureLogs

import ast
from textwrap import indent
import sys
import tempfile
from typing import Any
from copy import copy
from types import ModuleType

import numpy as np


# assign some dummy module that doesn't conflict
sys.modules["override_mod"] = np


@dataclass
class WrapScriptIntoMain:
    """
    Convenience utility: Sometimes you just want to profile/annotate a script once without modifying any source code.
    This makes a copy of the script file, wrapping its contents in a callable "main" function to which `maxray` transforms can then be applied.
    """

    script_path: str
    temp_sourcefile: Any = field(
        default_factory=lambda: tempfile.NamedTemporaryFile("w", delete=False)
    )

    def build(self):
        with open(self.script_path, "r") as file:
            source = file.read()

        new_source = f"""def main():
{indent(source, '    ')}
"""

        # inspect.getsource relies on the file actually existing
        with open(self.temp_sourcefile.name, "w") as f:
            f.write(new_source)

        tree = ast.parse(new_source, filename=self.temp_sourcefile.name)
        main_func = ast.fix_missing_locations(tree)

        # BUG: will error if `import *` is used (doesn't work inside a fn `def`)
        compiled_code = compile(
            main_func, filename=self.temp_sourcefile.name, mode="exec"
        )

        namespace = {}
        exec(compiled_code, namespace)
        # TODO: use non-conflicting name other than "main"
        main = namespace["main"]
        main.__module__ = "override_mod"
        return main

    def run(self):
        fn = xray(self.rewrite_node, initial_scope={"__name__": "__main__"})(
            self.build()
        )
        with CaptureLogs(self.script_path) as cl:
            fn()

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
        return x

    @staticmethod
    def empty_module(module_name: str):
        module_code = ""

        module_ast = ast.parse(module_code, filename=f"{module_name}.py")

        module_code_object = compile(
            module_ast, filename=f"{module_name}.py", mode="exec"
        )

        module = ModuleType(module_name)

        exec(module_code_object, module.__dict__)

        return module


def run_script():
    _set_logging(True)

    match sys.argv:
        case (
            _,
            script_path,
        ) if (
            path := Path(script_path).resolve(True)
        ).exists() and path.suffix == ".py":
            WrapScriptIntoMain(str(path)).run()

        case (_0, script_path, "--", *args):
            sys.argv = [_0, *args]
            path = Path(script_path).resolve(True)
            if path.exists() and path.suffix == ".py":
                WrapScriptIntoMain(str(path)).run()

        case _:
            raise RuntimeError(
                f"Incorrect argument usage - expected `capture-logs <script_path> -- script_args...` (got {sys.argv[1:]})"
            )
