import ast
import shutil
import subprocess
from importlib.util import find_spec
from pathlib import Path

import rich
import click

from typing import Literal


@click.group()
def cli():
    pass


class Builder(ast.NodeTransformer):
    def __init__(
        self,
        click_init: bool,
        include_runner: bool,
        use_rerun: bool,
        add_session: bool,
        inator_name: str,
    ):
        super().__init__()

        self.click_init = click_init
        self.include_runner = include_runner
        self.use_rerun = use_rerun
        self.add_session = add_session
        self.inator_name = inator_name

    def visit_Name(self, node: ast.Name):
        super().generic_visit(node)

        if node.id == "Inator":
            node.id = self.inator_name

        return node

    def visit_Import(self, node: ast.Import):
        match node.names:
            case [ast.alias(name="click")] if not self.click_init:
                return None
            case [ast.alias(name="rerun")] if not self.use_rerun:
                return None
            case _:
                return super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        match node.name:
            case "cli" if not self.click_init:
                return None
            case "runner" if not self.include_runner:
                return None
            case "match_run_result" if not self.include_runner:
                return None
            case "enter_session" if not self.add_session:
                return None
            case _:
                super().generic_visit(node)
                return node

    def visit_ClassDef(self, node: ast.ClassDef):
        super().generic_visit(node)
        node.name = self.inator_name
        return node


@cli.command()
@click.argument("path", type=Path)
@click.option("--new", "create_mode", flag_value="new", default=True)
@click.option("--over", "create_mode", flag_value="over")
@click.option("-f", "--force", is_flag=True, help="Overwrite existing file")
@click.option(
    "--cli",
    is_flag=True,
    help="Adds a static constructor that accepts CLI-style arguments",
)
@click.option(
    "--runner",
    is_flag=True,
    help="Adds manual control to spawn multiple runs of the program",
)
@click.option(
    "--rerun", is_flag=True, help="Imports rerun (requires the `rerun-sdk` package)"
)
@click.option(
    "--session",
    is_flag=True,
    help="A restricted version of `runner` that is just a contextmanager entered for the lifetime of the program. Useful for cleanup or displaying/saving results",
)
@click.option("--name", type=str, default="Inator", help="Name of the class to create")
def template(
    path: Path,
    create_mode: Literal["new", "over"],
    force: bool,
    cli: bool,
    runner: bool,
    rerun: bool,
    session: bool,
    name: str,
):
    save_path = path.resolve(False)

    if create_mode == "over":
        save_path = save_path.with_name(f"over_{save_path.name}")

    assert save_path.suffix == ".py"

    spec = find_spec("maxray.inators.template")
    assert spec is not None
    assert spec.origin is not None

    template_path = save_path
    if not force:
        assert not template_path.exists(), f"{template_path} exists!"

    source = Path(spec.origin).read_text()

    source_tree = ast.parse(source)

    visitor = Builder(
        click_init=cli,
        include_runner=runner,
        use_rerun=rerun,
        add_session=session,
        inator_name=name,
    )
    transformed_tree = visitor.visit(source_tree)

    transformed_source = ast.unparse(transformed_tree)

    template_path.write_text(transformed_source)

    # Format the code if `ruff` is in PATH
    if shutil.which("ruff") is not None:
        assert (
            template_path.is_file()
        )  # Don't want ruff to mess with all python files in a directory
        subprocess.check_output(["ruff", "format", str(template_path)])

    print(f"Wrote template to {template_path}")


@cli.command()
@click.argument("source_file", type=Path)
def ast_dump(source_file: Path):
    source = source_file.read_text()
    tree = ast.parse(source)
    rich.print(ast.dump(tree, indent=4))


def main():
    cli.main()


if __name__ == "__main__":
    main()
