from contextlib import ExitStack
from .reprs import structured_repr
from maxray import xray, _set_logging
from maxray.transforms import NodeContext

import pyarrow as pa
from pyarrow import feather as ft

from contextvars import ContextVar
from functools import wraps
import time
from pathlib import Path

from loguru import logger


class CaptureLogs:
    instance = ContextVar("CaptureLogs")

    ARROW_SCHEMA_FIELDS = {
        # Source location
        "loc_line_start": pa.int32(),
        "loc_line_end": pa.int32(),
        "loc_col_start": pa.int32(),
        "loc_col_end": pa.int32(),
        # Function calls
        "context_call_id": pa.string(),
        "target_call_id": pa.string(),
        "fn": pa.string(),
        "fn_call_count": pa.int32(),
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

    DEFAULT_LOG_FILE_NAME = ".maxray-logs.arrow"

    @staticmethod
    def extractor(x, ctx: NodeContext):
        if isinstance(instance := CaptureLogs.instance.get(None), CaptureLogs):
            if ctx.caller_id is not None:
                instance.fn_sources[ctx.caller_id] = ctx.fn_context
            if ctx.source != "self":
                instance.builder("loc_line_start").append(ctx.location[0])
                instance.builder("loc_line_end").append(ctx.location[1])
                instance.builder("loc_col_start").append(ctx.location[2])
                instance.builder("loc_col_end").append(ctx.location[3])

                instance.builder("context_call_id").append(ctx.fn_context.compile_id)
                instance.builder("target_call_id").append(ctx.caller_id)
                instance.builder("source_file").append(ctx.fn_context.source_file)
                instance.builder("source_module").append(ctx.fn_context.module)
                instance.builder("source").append(ctx.source)

                instance.builder("fn").append(ctx.fn_context.name)
                instance.builder("fn_call_count").append(
                    ctx.fn_context.call_count.get()
                )

                struct_repr = structured_repr(x)
                instance.builder("struct_repr").append(struct_repr)

                value_type = repr(type(x))
                instance.builder("value_type").append(value_type)

                type_name = type(x).__name__
                if type_name in CaptureLogs.LSP_NO_SHOW_TYPES:
                    instance.builder("lsp_repr").append(None)
                else:
                    instance.builder("lsp_repr").append(
                        f" {struct_repr}"
                    )  # space inserted for formatting

                instance.builder("timestamp").append(time.perf_counter())

                # Trigger flush here
                if len(instance.builder("timestamp")) > instance.flush_every_records:
                    instance.flush()

        return x

    @staticmethod
    def schema():
        return pa.schema(
            [pa.field(k, v) for k, v in CaptureLogs.ARROW_SCHEMA_FIELDS.items()]
        )

    def __init__(self, stream_to_arrow_file=None, flush_every_records: int = 10_000):
        # Maps function UUIDs (_MAXRAY_TRANSFORM_ID) to FnContext instances
        self.fn_sources = {}

        self.builders = {}

        if stream_to_arrow_file is not None:
            stream_to_arrow_file = Path(stream_to_arrow_file)
            if stream_to_arrow_file.exists():
                self.save_to = (
                    stream_to_arrow_file.resolve(True).parent
                    / self.DEFAULT_LOG_FILE_NAME
                )
            else:
                self.save_to = stream_to_arrow_file
        else:
            self.save_to = None

        # TODO: clearer separation of in-memory and write-to-sink modes
        self.in_memory_batches = []

        self.flush_every_records = flush_every_records

        self.write_context = ExitStack()

    def builder(self, name: str):
        if name not in self.builders:
            self.builders[name] = []

        return self.builders[name]

    def flush(self):
        if not self.builders:
            logger.warning("Nothing to flush")
            return

        # Because script may have been KeyboardInterrupt/SIGINT'd at any point, array lengths aren't guaranteed to be the same
        truncate_len = min(len(arr) for arr in self.builders.values())

        arrays, names = [], []
        for col_name, col_type in self.ARROW_SCHEMA_FIELDS.items():
            builder = self.builders[col_name]
            # builder = self.builders[col_name][:truncate_len] # blows up size +1GB memory / second
            arrays.append(pa.array(builder, type=col_type))
            names.append(col_name)

            builder.clear()

        batch = pa.RecordBatch.from_arrays(arrays=arrays, names=names)
        if self.save_to is not None:
            self.writer.write(batch)
        else:
            self.in_memory_batches.append(batch)
        return batch

    def __enter__(self):
        CaptureLogs.instance.set(self)

        # TODO: forbid re-entry

        if self.save_to is not None:
            self.sink = self.write_context.enter_context(
                pa.OSFile(str(self.save_to), "wb")
            )
            self.writer = self.write_context.enter_context(
                pa.ipc.new_file(self.sink, self.schema())
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                return

            CaptureLogs.instance.set(None)
        finally:
            self.flush()
            self.write_context.__exit__(exc_type, exc_val, exc_tb)

    def collect(self):
        return pa.Table.from_batches(self.in_memory_batches, schema=self.schema())

    @staticmethod
    def attach(*, debug: bool = False):
        def decor(f):
            fx = xray(CaptureLogs.extractor)(f)

            @wraps(f)
            def wrapper(*args, **kwargs):
                _set_logging(debug)
                return fx(*args, **kwargs)

            return wrapper

        return decor
