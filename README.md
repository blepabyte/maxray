# maxray

> the problem with doing weird metaprogramming shit is having to deal with *other people's weird metaprogramming shit*

Trace and modify the result of (almost) every single expression executed in a Python program. WIP.

```python
from maxray import transform, xray, maxray

from torch import tensor, Tensor, device


def move_tensor(x, ctx):
    if isinstance(x, Tensor) and x.device != device("cuda"):
        return x.to("cuda")
    return x


@transform(move_tensor)
def show_multiply(a, b):
    print(a @ b)

# Source code is rewritten to be equivalent to:

def _show_multiply(a, b):
    move_tensor(move_tensor(print, ...)(move_tensor(a, ...) @ move_tensor(b, ...)), ...)

# ---

show_multiply(
    tensor([[0.0, 1.0], [1.0, 1.0]], device="cpu"),
    tensor([[1.0], [1.0]], device="cuda"),
)  # Without the decorator, you'd expect `RuntimeError: Expected all tensors to be on the same device`

# tensor([[1.],
#         [2.]], device='cuda:0')
```

The `*xray` decorators will recursively trace and patch every single callable they encounter until reaching either builtins, native, or generated code.

The `ctx` argument contains context information about the location of the original source code, which may be useful to build editor/LSP integrations.

## Installation

```sh
pip install maxray
```

Actively supported Python versions:
- 3.11
- 3.12
