from __future__ import annotations

import shlex
import click

import sys
from pydoc import importfile
import ast
import importlib.util
from result import Result, Ok, Err
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Callable


@dataclass
class ScriptFile:
    path: Path

    def evaluated(self):
        prev_sys_path = [*sys.path]
        sys.path.append(str(self.path.parent.resolve(True)))
        try:
            return vars(importfile(str(self.path)))
        finally:
            sys.path = prev_sys_path


@dataclass
class PackageModule:
    module_name: str
    module_spec: Any

    def evaluated(self):
        if self.module_name in sys.modules:
            # TODO: handle failed reloads
            module = importlib.reload(sys.modules[self.module_name])
        else:
            module = importlib.import_module(self.module_name)
        return vars(module)


@dataclass
class CodeString:
    code: str

    def evaluated(self):
        # TODO: put extra convenience stuff into scope?
        return {"<lambda>": eval(self.code)}


@dataclass
class ReloadableFunction:
    source: CodeString | ScriptFile | PackageModule
    function_name: str
    function: Callable
    depends_on_files: frozenset[str]
    depends_on_dirs: frozenset[str]

    def triggers_reload(self, changed_files: set[str]):
        # TODO: dir check
        return len(self.depends_on_files.intersection(changed_files)) > 0

    def reload(self) -> Result:
        try:
            self.function = self.source.evaluated()[self.function_name]
            return Ok(self.function)
        except Exception as e:
            return Err(f"{e}: Function reload failed")

    def call(self, x, ray):
        return self.function(x, ray)


@dataclass
class ReloadableInstance:
    source: ScriptFile | PackageModule
    type_name: str
    instance: Any
    depends_on_files: frozenset[Path]
    depends_on_dirs: frozenset[Path]

    def triggers_reload(self, changed_files):
        # TODO: dir check
        return len(self.depends_on_files.intersection(changed_files))

    def reload(self) -> Result:
        with self.instance._handle_reload():
            new_instance_type = self.source.evaluated()[self.type_name]
            self.instance.__class__ = new_instance_type
            return Ok(self.instance)
        return Err("Instance reload failed")

    def call(self, x, ray):
        return self.instance(x, ray)


def reloadable_from_spec(
    spec: str,
) -> Result[ReloadableFunction | ReloadableInstance, str]:
    if spec.startswith("lambda "):  # Cannot be reloaded
        source = CodeString(spec)
        try:
            function = source.evaluated()["<lambda>"]
        except Exception as e:
            return Err(f"{e}: Failed evaluating <lambda> function definition")
        return Ok(
            ReloadableFunction(source, "<lambda>", function, frozenset(), frozenset())
        )

    match spec.split(" ", maxsplit=1):
        case [spec_location, spec_args]:
            pass
        case [spec_location]:
            spec_args = ""
        case _:
            return Err(f"Too many spaces: {spec}")

    match spec_location.rsplit(":", maxsplit=1):
        case [spec_source, spec_symbol]:
            pass
        case _:
            return Err(f"Missing symbol specification: {spec}")

    depends_on_paths = []

    try:
        module_spec = importlib.util.find_spec(spec_source)
    except (ModuleNotFoundError, ImportError):
        module_spec = None
    if module_spec is None:
        try:
            # Shortcut for included inators
            module_spec = importlib.util.find_spec(f"maxray.inators.{spec_source}")
            spec_source = f"maxray.inators.{spec_source}"
        except (ModuleNotFoundError, ImportError):
            pass

    if module_spec is not None and module_spec.origin is not None:
        source = PackageModule(spec_source, module_spec)
        # TODO: extract all files of submodule as dependents
        try:
            module = source.evaluated()
        except Exception as e:
            return Err(f"{e}: Failed loading module {spec_source}")

        try:
            symbol = module[spec_symbol]
        except KeyError:
            return Err(f"Symbol {spec_symbol} was not defined in module {spec_source}")

        depends_on_paths.append(Path(module_spec.origin).resolve(True))

    elif (source_file := Path(spec_source)).exists():
        source = ScriptFile(source_file)
        try:
            file = source.evaluated()
        except Exception as e:
            return Err(f"{e}: Failed running script {spec_source}")

        try:
            symbol = file[spec_symbol]
        except KeyError:
            return Err(f"Symbol {spec_symbol} was not defined in {spec_source}")

        depends_on_paths.append(source_file.resolve(True))

    else:
        return Err(f"Source {spec_source} is neither a file nor module")

    if isinstance(symbol, type):
        match getattr(symbol, "cli", None):
            case click.Command() as command:
                cli_context = command.make_context(spec_symbol, shlex.split(spec_args))
                new_instance = symbol.cli.invoke(cli_context)
            case None:
                new_instance = symbol()
            case _:
                return Err("Expected `cli` to be a click Command")

        return Ok(
            ReloadableInstance(
                source,
                spec_symbol,
                new_instance,
                frozenset(depends_on_paths),
                frozenset(),
            )
        )
    elif callable(symbol):
        return Ok(
            ReloadableFunction(
                source, spec_symbol, symbol, frozenset(depends_on_paths), frozenset()
            )
        )
    else:
        return Err(f"Loaded symbol from {source} is neither a type nor callable")
