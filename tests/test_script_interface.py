from maxray.capture import ScriptRunner

from pathlib import Path


def nonzero(x, ctx):
    if isinstance(x, int) and x == 0:
        return 1
    return x


def test_sample_script():
    result_fail = ScriptRunner.run_script(Path(__file__).parent / "sample_script.py")
    assert not result_fail.completed()

    result_ok = ScriptRunner.run_script(
        Path(__file__).parent / "sample_script.py", with_decor_inator=nonzero
    )
    assert result_ok.completed()


def test_sample_module():
    results = ScriptRunner.run_module("maxray._test_module")
    assert (
        results.functions_arrow.to_pandas()
        .source_file.str.contains("_test_module")
        .any()
    )
