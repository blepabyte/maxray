from maxray.inators.core import S, Ray
from maxray.inators.base import BaseInator
from maxray.runner import RunCompleted, RunErrored
from maxray.reprs import structured_repr
from maxray.function_store import FunctionStore

from result import Result, Ok, Err
import time
from uuid import uuid4

import rerun as rr


def build_source_block(x, ray: Ray) -> Result:
    fn_context = ray.ctx.fn_context

    # TODO: can cache this lookup
    source_offset_lines = FunctionStore.get(
        fn_context.compile_id
    ).data.source_offset_lines

    if source_offset_lines is None:
        return Err("missing source_offset_lines")

    function_lines = fn_context.source.splitlines()

    output_lines = [
        f"{fn_context.module}",
        f"{fn_context.source_file}",
    ]
    output_lines.append("```python")

    up_to_including_active_line = ray.ctx.location[1] - source_offset_lines
    output_lines.extend(function_lines[: up_to_including_active_line + 1])
    num_hash_pads = len(output_lines[-1]) + 3
    output_lines[-1] += f"  # <-- 💥 u are here ({ray.ctx.source})"
    output_lines.append("#" * num_hash_pads)
    output_lines.append("```\n")

    output_lines.append("```")

    # Attempt at avoiding repr of incompletely initialised objects
    if "__init__" not in fn_context.name:
        try:
            x_repr = repr(x)
        except Exception as e:
            x_repr = f"! unrepresentable: {e}"
    else:
        x_repr = f"{type(x).__name__} @ {id(x)}"

    output_lines.append(x_repr)
    output_lines.append("```\n")

    output_lines.append("```python")
    output_lines.extend(function_lines[up_to_including_active_line + 1 :])
    output_lines.append("```")

    return Ok("\n".join(output_lines))


class Source(BaseInator):
    def __init__(self):
        super().__init__()
        self.current_function = None
        self.current_function_logs = []

    def xray(self, x, ray: Ray):
        S.define_once(
            "RERUN_INSTANCE",
            lambda _: rr.init(f"{self}", spawn=True, recording_id=str(uuid4())),
            v=1,
        )

        if isinstance(x, (RunCompleted, RunErrored)):
            return

        match build_source_block(x, ray):
            case Ok(annotated_function_source):
                markdown_blocks = f"""{annotated_function_source}"""
                rr.log(
                    "current_function",
                    rr.TextDocument(markdown_blocks, media_type="text/markdown"),
                )
            case Err(e):
                rr.log(
                    "current_function",
                    rr.TextDocument(
                        f"{e}: Err building function source representation",
                        media_type="text/markdown",
                    ),
                )
