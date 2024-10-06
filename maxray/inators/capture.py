from maxray.inators.core import R, S, NodeContext, Ray
from maxray.inators.base import BaseInator
from maxray.runner import (
    MAIN_FN_NAME,
    ExecInfo,
    RunAborted,
    RunCompleted,
    RunErrored,
    AbortRun,
    RestartRun,
    Break,
)
from maxray.reprs import structured_repr
from maxray.function_store import FunctionStore

import pyarrow as pa
import pyarrow.compute as pc

import time
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from contextlib import contextmanager

from typing import Optional

import click


class CaptureLogs(BaseInator):
    def __init__(self, flush_every_records: int, flush_every_seconds: float):
        self.flush_every_records = flush_every_records
        self.flush_every_seconds = flush_every_seconds

        self.builders = {}

        self.last_flush_time = time.perf_counter()

    def xray(self, x, ray: Ray):
        ctx: NodeContext = ray.ctx

        # TODO: make these fields more extensible
        self.builder("loc_line_start").append(ctx.location[0])
        self.builder("loc_line_end").append(ctx.location[1])
        self.builder("loc_col_start").append(ctx.location[2])
        self.builder("loc_col_end").append(ctx.location[3])

        self.builder("fn_compile_id").append(ctx.fn_context.compile_id)
        self.builder("source_file").append(ctx.fn_context.source_file)
        self.builder("source_module").append(ctx.fn_context.module)
        self.builder("source").append(ctx.source)

        self.builder("fn").append(ctx.fn_context.name)

        struct_repr = structured_repr(x)
        self.builder("struct_repr").append(struct_repr)

        value_type = repr(type(x))
        self.builder("value_type").append(value_type)

        type_name = type(x).__name__
        if type_name in CaptureLogs.LSP_NO_SHOW_TYPES:
            self.builder("lsp_repr").append(None)
        else:
            self.builder("lsp_repr").append(
                f" {struct_repr}"
            )  # space inserted for formatting

        self.builder("timestamp").append(time.perf_counter())

        flush_triggered_by_time = (
            time.perf_counter() - self.last_flush_time > self.flush_every_seconds
        )
        flush_triggered_by_count = (
            num_records := len(self.builder("timestamp"))
        ) > self.flush_every_records
        if flush_triggered_by_time or flush_triggered_by_count:
            if num_records > 0:
                self.flush()
            else:
                ray.log("No records to flush", level="WARN")
            self.last_flush_time = time.perf_counter()

    def builder(self, name: str):
        if name not in self.builders:
            self.builders[name] = []

        return self.builders[name]

    @staticmethod
    def schema():
        return pa.schema(
            [pa.field(k, v) for k, v in CaptureLogs.ARROW_SCHEMA_FIELDS.items()]
        )

    ARROW_SCHEMA_FIELDS = {
        # Source location
        "loc_line_start": pa.int32(),
        "loc_line_end": pa.int32(),
        "loc_col_start": pa.int32(),
        "loc_col_end": pa.int32(),
        # Function calls
        "fn_compile_id": pa.string(),
        "fn": pa.string(),
        # Source info/code
        "source_file": pa.string(),
        "source_module": pa.string(),
        "source": pa.string(),
        # Extracted data
        "struct_repr": pa.string(),
        "lsp_repr": pa.string(),
        "value_type": pa.string(),
        "timestamp": pa.float64(),
    }

    LSP_NO_SHOW_TYPES = {
        "type",
        "function",
        "builtin_function_or_method",
        "cython_function_or_method",
        "staticmethod",
        "method",
        "module",
        "NoneType",
        "int",  # mostly uninteresting and makes array indexing expressions cluttered
        "int32",
        "int64",
    }

    def flush(self):
        num_records = len(self.builder("timestamp"))
        if num_records == 0:
            return

        # TODO: Because script may have been KeyboardInterrupt/SIGINT'd at any point, array lengths aren't guaranteed to be the same
        truncate_len = min(len(arr) for arr in self.builders.values())

        arrays, names = [], []
        for col_name, col_type in self.ARROW_SCHEMA_FIELDS.items():
            builder = self.builders[col_name]
            # builder = self.builders[col_name][:truncate_len] # blows up size +1GB memory / second
            arrays.append(pa.array(builder, type=col_type))
            names.append(col_name)

            builder.clear()

        batch = pa.RecordBatch.from_arrays(arrays=arrays, schema=self.schema())
        R.CaptureLogBatches.push(batch)
        return batch


class Write(CaptureLogs):
    """
    Write batches to a file as they arrive so they do not need to be held in-memory.
    """

    @dataclass
    class SaveAbsolutePath:
        path: Path

    @dataclass
    class SaveRelativeToScript:
        name: str

    def __init__(self, save_location: SaveRelativeToScript | SaveAbsolutePath):
        # No point in fast updates
        super().__init__(flush_every_records=10_000, flush_every_seconds=float("inf"))

        self.save_location = save_location

    @staticmethod
    @click.command()
    @click.option("-o", "--output", type=Path)
    @click.option("--name", type=str, default=".maxray-logs.arrow")
    def cli(output: Optional[Path], name: str):
        if output is None:
            return Write(Write.SaveRelativeToScript(name=name))
        else:
            return Write(Write.SaveAbsolutePath(path=output))

    def write_batches(self, writer, iter_batches):
        for batch in iter_batches:
            writer.write_batch(batch)
            yield

    @contextmanager
    def enter_session(self, xi: ExecInfo):
        match self.save_location:
            case Write.SaveRelativeToScript(name=name):
                if xi.source_origin is None:
                    write_from = Path(".")
                else:
                    write_from = xi.source_origin.parent

                write_to_arrow_file = write_from / name

            case Write.SaveAbsolutePath(path=write_to_arrow_file):
                ...

        with (
            super().enter_session(xi),
            pa.OSFile(str(write_to_arrow_file), "wb") as sink,
            pa.ipc.new_file(sink, self.schema()) as writer,
            R.CaptureLogBatches.stack(partial(self.write_batches, writer)),
        ):
            try:
                yield
            finally:
                self.flush()


class Collect(CaptureLogs):
    """
    Expose a `collect` method to be implemented by subclasses, called once at the end of the session, given tables containing all logs and functions.
    """

    def __init__(self):
        super().__init__(flush_every_records=10_000, flush_every_seconds=float("inf"))

        self.in_memory_batches = []

    def collect_batches(self, iter_batches):
        for batch in iter_batches:
            self.in_memory_batches.append(batch)
            yield

    def collect(self, logs: pa.Table, funs: pa.Table):
        raise NotImplementedError()

    @contextmanager
    def enter_session(self, xi: ExecInfo):
        try:
            with (
                super().enter_session(xi),
                R.CaptureLogBatches.stack(self.collect_batches),
            ):
                try:
                    yield
                finally:
                    self.flush()

        finally:
            logs = pa.Table.from_batches(self.in_memory_batches, schema=self.schema())
            funs = FunctionStore.collect()
            if xi.source_origin is not None:
                funs = patch_function_source_files(
                    funs, xi.temp_exec_file, xi.source_origin
                )
            self.collect(logs, funs)


def patch_function_source_files(funs: pa.Table, temp_source, run_source):
    # Patch the correct source file names (temporary -> actual)
    sf_col_idx = funs.column_names.index("source_file")
    # BUG: this would probably fail badly if the file name contains special regex chars...
    remapped_source_file = pc.replace_substring_regex(
        funs["source_file"],
        pattern=rf"^{temp_source}$",
        replacement=f"{run_source}",
    )
    funs = funs.set_column(sf_col_idx, funs.field(sf_col_idx), remapped_source_file)
    return funs
