from maxray.inators.core import S, Ray
from maxray.inators.base import BaseInator
from maxray.runner import RunCompleted, RunErrored
from maxray.reprs import structured_repr

import time
from uuid import uuid4

import rerun as rr


def value_repr(x):
    if isinstance(x, (str, int, float, bool, type)):
        return str(x)
    # TODO
    if isinstance(x, list):
        contents = (value_repr(item) for item in x[:3])
        return "[" + ", ".join(contents) + "]"
    if x is None:
        return "None"
    return "¿¿¿"


class Source(BaseInator):
    def __init__(self):
        super().__init__()
        self.current_function = None
        self.current_function_logs = []

    def xray(self, x, ray: Ray):
        ctx = ray.ctx

        S.define_once(
            "RERUN_INSTANCE",
            lambda _: rr.init(f"{self}", spawn=True, recording_id=str(uuid4())),
            v=1,
        )

        if isinstance(x, (RunCompleted, RunErrored)):
            return

        # if isinstance(x, str):
        #     rr.log("strings", rr.TextLog(x))

        fn_source = ctx.fn_context.source

        fn_lines = fn_source.splitlines()

        try:
            expr_line = ctx.location[0] - ctx.fn_context.line_offset + 1
            fn_lines[expr_line] = f"{fn_lines[expr_line]}"
            expr_repr = (
                f"```\n{ctx.source}: {structured_repr(x)} = {value_repr(x)}\n```"
            )

            repr_pre = "\n".join(fn_lines[:expr_line])
            repr_post = "\n".join(fn_lines[expr_line + 1 :])
            repr_state = f"""
```
{ctx.fn_context.name} in module {ctx.fn_context.module}
{ctx.fn_context.source_file}
```

```python
{repr_pre}
```
---

{expr_repr}

```python
{fn_lines[expr_line]}
```

---

```python
{repr_post}
```
"""
        except Exception as e:
            rr.log(
                "LiveGrep:internal",
                rr.TextLog(
                    f"{e}: {type(e).__qualname__}", level=rr.TextLogLevel("ERROR")
                ),
            )
            return

        rr.log(
            "current_function", rr.TextDocument(repr_state, media_type="text/markdown")
        )
