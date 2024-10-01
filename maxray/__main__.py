import re
from importlib.util import find_spec
from pathlib import Path

import click

from typing import Optional


@click.group()
def cli():
    pass


@cli.command()
@click.option("--over", type=str)
@click.option("--new", type=str)
@click.option("-f", "--force", is_flag=True)
@click.option("--runner", is_flag=True)
def template(over: Optional[str], new: Optional[str], force: bool, runner: bool):
    save_path_args = [f for f in [over, new] if f is not None]

    if not save_path_args or len(save_path_args) > 1:
        raise ValueError("Must specify exactly one of --new or --over")

    save_path = Path(save_path_args[0]).resolve(False)
    if over is not None:
        save_path = save_path.with_name(f"over_{save_path.name}")

    assert save_path.suffix == ".py"

    spec = find_spec("maxray.inators.template")
    assert spec is not None
    assert spec.origin is not None

    template_path = save_path
    if not force:
        assert not template_path.exists(), f"{template_path} exists!"

    source = Path(spec.origin).read_text()
    if not runner:
        source = re.sub(r"\s+def runner(?:\n|.*)+", "", source, flags=re.MULTILINE)
        source += "\n"
    template_path.write_text(source)
    print(f"Wrote template to {template_path}")


def main():
    cli.main()


if __name__ == "__main__":
    main()
