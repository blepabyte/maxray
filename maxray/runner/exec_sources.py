from __future__ import annotations

import sys
from pydoc import importfile
import ast
import importlib.util
from result import Result, Ok, Err
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any


@dataclass
class ScriptFile:
    path: Path

    def evaluated(self):
        return importfile(str(self.path))


@dataclass
class PackageModule:
    module_name: str
    module_spec: Any

    def evaluated(self):
        if self.module_name in sys.modules:
            try:
                module = importlib.reload(sys.modules[self.module_name])
            except Exception as e:
                print(e)
                return None
        else:
            module = importlib.import_module(self.module_name)
        return module


@dataclass
class CodeString:
    code: str

    def evaluated(self):
        # TODO: Put everything in sys.modules (or at least the main script file?) into the eval scope
        return eval(self.code)


class ExecSource:
    def __init__(self, source: ScriptFile | PackageModule | CodeString):
        self.source = source
        self.affected_by_source_files = set()

    @staticmethod
    def new(maybe_source: str) -> Result[ExecSource, str]:
        """
        Tries to auto-detect the type of source
        """
        return (
            ExecSource.new_script(maybe_source)
            .or_else(lambda _: ExecSource.new_module(maybe_source))
            .or_else(lambda _: ExecSource.new_code(maybe_source))
        )

    @staticmethod
    def new_script(script_path) -> Result[ExecSource, str]:
        if (file := Path(script_path)).is_file() and file.suffix == ".py":
            file = file.resolve()

            new = ExecSource(ScriptFile(file))
            return Ok(new.add_source_effects([file]))
        else:
            return Err(f"{script_path}: not a file")

    @staticmethod
    def new_module(module_name) -> Result[ExecSource, str]:
        try:
            module_spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            return Err(f"{module_name}: module doesn't exist")

        if module_spec is None or module_spec.origin is None:
            return Err(f"{module_name}: module doesn't seem to exist")

        new = ExecSource(PackageModule(module_name, module_spec))
        return Ok(new.add_source_effects([module_spec.origin]))

    @staticmethod
    def new_code(source_code) -> Result[ExecSource, str]:
        try:
            ast.parse(source_code)
        except SyntaxError as err:
            return Err(f"{err}: invalid syntax for code source ({source_code})")

        return Ok(ExecSource(CodeString(source_code)))

    def add_source_effects(self, paths: list):
        self.affected_by_source_files.update(
            {str(Path(p).resolve(True)) for p in paths}
        )
        return self


@dataclass
class DynamicSymbol:
    source: ExecSource
    symbol: Optional[str]

    def load(self):
        mod = self.source.source.evaluated()
        if self.symbol is not None:
            return getattr(mod, self.symbol)
        else:
            return mod

    def is_changed_by_files(self, updated_files: set[str]) -> bool:
        """
        Paths represented by absolute strings (which are what `watchfiles` gives as changes)
        """
        return len(self.source.affected_by_source_files & updated_files) > 0

    def files_to_watch(self):
        return list(self.source.affected_by_source_files)

    @staticmethod
    def from_spec(spec: str, fallback_source_spec=None) -> Result:
        if spec.startswith("lambda "):
            return ExecSource.new_code(spec).map(
                lambda source: DynamicSymbol(source, None)
            )

        match spec.rsplit(":", maxsplit=1):
            case [source_spec, symbol]:
                return ExecSource.new(source_spec).map(
                    lambda source: DynamicSymbol(source, symbol)
                )
            case [symbol] if fallback_source_spec is not None:
                return ExecSource.new(fallback_source_spec).map(
                    lambda source: DynamicSymbol(source, symbol)
                )
            case _:
                return Err(f"{spec} is missing a valid separator")
