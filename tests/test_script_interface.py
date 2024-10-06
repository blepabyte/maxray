from maxray.nodes import RayContext
from maxray.runner import ScriptRunner, RunCompleted, RunErrored

from pathlib import Path


def nonzero(x, ray: RayContext):
    if ray.ctx.id != "constant" and isinstance(x, int) and x == 0:
        return 1
    return x


def test_sample_script():
    result_fail = ScriptRunner.run_script(Path(__file__).parent / "sample_script.py")
    assert isinstance(result_fail, RunErrored)

    result_ok = ScriptRunner.run_script(
        Path(__file__).parent / "sample_script.py", with_decor_inator=nonzero
    )
    assert isinstance(result_ok, RunCompleted)


def test_sample_module():
    results = ScriptRunner.run_module("maxray._test_module")
    assert isinstance(results, RunCompleted)
