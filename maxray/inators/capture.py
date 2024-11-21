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
import pyarrow.feather as ft
import pyarrow.compute as pc

import time
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from contextlib import contextmanager, ExitStack

from typing import Optional

import click


class CaptureLogs(BaseInator):
    """
    Exposes log stream via `R.CaptureLogBatches`. The output/schema format can be customised/overriden by implementing:
    - `build_record`
    - `schema_fields`
    - `xray` (optional) (but make sure to call `super().xray(x, ray)`)
    """

    def __init__(self, flush_every_records: int, flush_every_seconds: float):
        super().__init__()
        self.flush_every_records = flush_every_records
        self.flush_every_seconds = flush_every_seconds

        self.builders = {}

        self.last_flush_time = time.perf_counter()

    def build_record(self, x, ray: Ray):
        ctx: NodeContext = ray.ctx

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

        self.builder("timestamp").append(time.time())

    def xray(self, x, ray: Ray):
        self.build_record(x, ray)

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
                ray.log("No records to flush", level="WARNING")
            self.last_flush_time = time.perf_counter()

    def builder(self, name: str):
        if name not in self.builders:
            self.builders[name] = []

        return self.builders[name]

    def schema(self):
        return pa.schema([pa.field(k, v) for k, v in self.schema_fields().items()])

    def schema_fields(self):
        return {
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
            "value_type": pa.string(),
            "timestamp": pa.float64(),
        }

    def flush(self):
        num_records = len(self.builder("timestamp"))
        if num_records == 0:
            return

        # TODO: Because script may have been KeyboardInterrupt/SIGINT'd at any point, array lengths aren't guaranteed to be the same
        truncate_len = min(len(arr) for arr in self.builders.values())

        arrays, names = [], []
        for col_name, col_type in self.schema_fields().items():
            builder = self.builders[col_name]
            # builder = self.builders[col_name][:truncate_len] # blows up size +1GB memory / second
            arrays.append(pa.array(builder, type=col_type))
            names.append(col_name)

            builder.clear()

        batch = pa.RecordBatch.from_arrays(arrays=arrays, schema=self.schema())
        R.CaptureLogBatches.push(batch)
        return batch

    @contextmanager
    def enter_session(self, xi: ExecInfo):
        with super().enter_session(xi):
            try:
                yield
            finally:
                self.flush()


class Write(CaptureLogs):
    """
    Write batches to a file as they arrive so they do not need to be held in-memory.
    """

    def __init__(self, save_logs: Optional[Path], save_functions: Optional[Path]):
        # No point in fast updates
        super().__init__(flush_every_records=10_000, flush_every_seconds=float("inf"))

        self.save_logs = save_logs
        self.save_functions = save_functions

    @staticmethod
    @click.command()
    @click.option("--logs", type=Path)
    @click.option("--functions", type=Path)
    def cli(logs: Optional[Path], functions: Optional[Path]):
        return Write(save_logs=logs, save_functions=functions)

    def write_log_batches(self, writer, iter_batches):
        for batch in iter_batches:
            writer.write_batch(batch)
            yield

    @contextmanager
    def enter_session(self, xi: ExecInfo):
        with (
            super().enter_session(xi) as _,
            ExitStack() as stack,
        ):
            if self.save_logs is not None:
                sink = stack.enter_context(pa.OSFile(str(self.save_logs), "wb"))
                writer = stack.enter_context(pa.ipc.new_file(sink, self.schema()))
                stack.enter_context(
                    R.CaptureLogBatches.stack(partial(self.write_log_batches, writer))
                )
            try:
                yield
            finally:
                self.flush()

                if self.save_functions is not None:
                    funs = FunctionStore.collect()
                    ft.write_feather(funs, str(self.save_functions))


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
            self.collect(logs, funs)
