import re
from importlib.util import find_spec
from pathlib import Path

import click


@click.group()
def cli():
    pass


@cli.command()
@click.argument("file", type=str)
@click.option("-f", "--force", is_flag=True)
@click.option("--runner", is_flag=True)
def template(file: str, force: bool, runner: bool):
    path = Path(file).resolve(True)
    assert path.suffix == ".py"

    spec = find_spec("maxray.inators.template")
    assert spec is not None
    assert spec.origin is not None

    template_path = path.with_name(f"over_{path.name}")
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
