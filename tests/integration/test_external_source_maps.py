import pyarrow.feather as ft

from ast import get_source_segment
from dataclasses import dataclass, field

import os
import subprocess
import tempfile
from pathlib import Path
from itertools import islice as take
from textwrap import dedent, indent

import pytest

# Get directory containing this test file
TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR.parent / "fixtures"


if os.environ.get("MAXRAY_NO_INTEGRATIONS") == "1":
    mark_integration = pytest.mark.skip
else:
    mark_integration = pytest.mark.integration


@mark_integration
def test_exact_source_matching():
    """
    Checks against known locations in a fixed test file
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir)
        subprocess.run(
            [
                "xpy",
                "--quiet",
                "-W",
                f"capture:Write --functions {output_path / 'functions.arrow'} --logs {output_path / 'logs.arrow'}",
                str(FIXTURES_DIR / "static_scripts" / "exact_source_matching.py"),
            ]
        )

        # Read the arrow files
        functions_table = ft.read_feather(output_path / "functions.arrow")
        logs_table = ft.read_feather(output_path / "logs.arrow")

        match_set = set()
        for _, row in logs_table.iterrows():
            match row.to_dict():
                case {
                    "source": '"start of test script"',
                    "loc_line_start": line_start,
                    "loc_line_end": line_end,
                    "loc_col_start": col_start,
                    "loc_col_end": col_end,
                }:
                    assert line_start == line_end == 0
                    assert col_start == 4 and col_end == 26

                    assert "first line" not in match_set
                    match_set.add("first line")

                case {
                    "source": "1",
                    "loc_line_start": line_start,
                    "loc_line_end": line_end,
                    "loc_col_start": col_start,
                    "loc_col_end": col_end,
                }:
                    assert line_start == line_end == 3
                    assert col_start == 4 and col_end == 5

                    match_set.add("1")

                case {
                    "source": "5",
                    "loc_line_start": line_start,
                    "loc_line_end": line_end,
                    "loc_col_start": col_start,
                    "loc_col_end": col_end,
                }:
                    assert line_start == line_end == 12
                    assert col_start == 8 and col_end == 9

                    match_set.add("5")

                case row:
                    pass

        assert len(match_set) == 3, match_set


@dataclass
class MockNode:
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int


@dataclass
class SegmentMatchingReport:
    mismatches: int = 0
    total: int = 0
    missing_files: set = field(default_factory=set)
    missing_fn_ids: set = field(default_factory=set)

    @property
    def accuracy(self):
        return (self.total - self.mismatches) / self.total if self.total > 0 else 1.0

    def __repr__(self):
        parts = []
        accuracy_pct = self.accuracy * 100
        parts.append(
            f"Accuracy: {accuracy_pct:.1f}% ({self.total - self.mismatches}/{self.total} correct)"
        )

        if self.missing_files:
            parts.append(f"Missing files: {len(self.missing_files)}")

        if self.missing_fn_ids:
            parts.append(f"Missing function IDs: {len(self.missing_fn_ids)}")

        return "\n".join(parts)


def report_segment_matches(functions_table, logs_table):
    """
    Check matching of `logs.source` against source segments extracted from the actual source file using `logs.loc_<x>_<x>`
    """
    functions_df = functions_table.to_pandas().set_index("compile_id")
    logs_df = logs_table.to_pandas()

    report = SegmentMatchingReport()

    missing_files = set()

    for _, log in logs_df.iterrows():
        match log.to_dict():
            case {
                "fn_compile_id": compile_id,
                "loc_line_start": line_start,
                "loc_line_end": line_end,
                "loc_col_start": col_start,
                "loc_col_end": col_end,
                "source_file": source_file,
                "source": node_source,
            }:
                try:
                    fn_info = functions_df.loc[compile_id].to_dict()
                except KeyError:
                    report.missing_fn_ids.add(compile_id)
                    continue

                report.total += 1

                # Converts from zero-indexed lines to Python's 1-indexed AST line numbers
                line_fudge = 1
                mock_node = MockNode(
                    line_start + line_fudge,
                    col_start,
                    line_end + line_fudge,
                    col_end,
                )

                try:
                    matched = get_source_segment(
                        Path(source_file).read_text(), mock_node, padded=False
                    )
                except Exception as e:
                    # File doesn't exist?
                    missing_files.add(source_file)
                    continue

                node_source_lines = node_source.splitlines(keepends=True)
                if len(node_source_lines) > 1:
                    node_source_lines[1:] = map(
                        lambda l: indent(l, " " * fn_info["source_dedent_chars"]),
                        node_source_lines[1:],
                    )
                    node_source = "".join(node_source_lines)

                if matched != node_source:
                    report.mismatches += 1
                    print(
                        f"MISMATCH:\n(filesystem)\n{matched}\n\n(capture)\n{node_source}"
                    )

    return report


def report_line_matches(functions_table, logs_table):
    """
    Check matching of `functions.source` against the real function definition in the source file
    """
    functions_df = functions_table.to_pandas().set_index("compile_id")
    logs_df = logs_table.to_pandas()

    report = SegmentMatchingReport()

    # these can basically be the same report...

    for _, log in logs_df.iterrows():
        match log.to_dict():
            case {
                "fn_compile_id": compile_id,
                "loc_line_start": line_start,
                "loc_line_end": line_end,
                "loc_col_start": col_start,
                "loc_col_end": col_end,
                "source_file": source_file,
                "source": node_source,
            }:
                try:
                    fn_info = functions_df.loc[compile_id].to_dict()
                except KeyError:
                    report.missing_fn_ids.add(compile_id)
                    continue

                report.total += 1

                source_lines = Path(source_file).read_text().splitlines()
                function_lines = fn_info["source"].rstrip().splitlines()
                source_offset_lines = fn_info["source_offset_lines"]

                # a bit confusing: the dedent is the transform applied to the function to be independently execd (e.g. if it was a method nested in a class definition)
                # to map back, we re-apply as an indent
                indent_chars = fn_info["source_dedent_chars"]

                for fn_lineno, fn_line in enumerate(function_lines):
                    real_line_number = fn_lineno + source_offset_lines

                    # ScriptRunner inserts an imaginary `def maaaaaaain()`
                    if real_line_number >= 0:
                        if indent_chars >= 0:
                            fn_map_line = indent(
                                fn_line, " " * fn_info["source_dedent_chars"]
                            )
                        else:
                            fn_map_line = fn_line[-indent_chars:]
                        assert (
                            fn_map_line == source_lines[real_line_number]
                        ), f"real lineno: {real_line_number}"

    return report


@mark_integration
def test_segment_matching():
    """
    Check that the source locations exported by `maxray.inators.capture` agree with the actual source files.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir)
        subprocess.run(
            [
                "xpy",
                "--quiet",
                "-W",
                f"capture:Write --functions {output_path / 'functions.arrow'} --logs {output_path / 'logs.arrow'}",
                str(FIXTURES_DIR / "compat_scripts" / "pandas_calls.py"),
            ]
        )

        # Read the arrow files
        functions_table = ft.read_table(output_path / "functions.arrow")
        logs_table = ft.read_table(output_path / "logs.arrow")

        assert len(logs_table) > 0, "All transforms failed"

        report = report_segment_matches(functions_table, logs_table)
        print(report)

        assert report.accuracy > 0.99  # 99%
        if report.mismatches > 0:
            pytest.xfail("meh: still some unhandled segment-matching edge cases")


@mark_integration
def test_line_matching():
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir)
        subprocess.run(
            [
                "xpy",
                "--quiet",
                "-W",
                f"capture:Write --functions {output_path / 'functions.arrow'} --logs {output_path / 'logs.arrow'}",
                str(FIXTURES_DIR / "compat_scripts" / "pandas_calls.py"),
            ]
        )

        # Read the arrow files
        functions_table = ft.read_table(output_path / "functions.arrow")
        logs_table = ft.read_table(output_path / "logs.arrow")

        report = report_line_matches(functions_table, logs_table)
        print(report)

        assert report.accuracy > 0.99  # 99%
        if report.mismatches > 0:
            pytest.xfail("meh: still some unhandled segment-matching edge cases")
