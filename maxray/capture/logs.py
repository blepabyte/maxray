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

    LSP_NO_SHOW_TYPES = {
        "type",
        "function",
        "builtin_function_or_method",
        "staticmethod",
        "method",
        "module",
        "NoneType",
        "int",  # mostly uninteresting and makes array indexing expressions cluttered
        "int32",
        "int64",
    }

    @staticmethod
    def extractor(x, ctx: NodeContext):
        if isinstance(instance := CaptureLogs.instance.get(None), CaptureLogs):
            if ctx.caller_id is not None:
                instance.fn_sources[ctx.caller_id] = ctx.fn_context
            if ctx.source != "self":
                parent_fn_id = getattr(
                    ctx.fn_context.impl_fn, "_MAXRAY_TRANSFORM_ID", None
                )

                instance.builder("loc_line_start").append(ctx.location[0])
                instance.builder("loc_line_end").append(ctx.location[1])
                instance.builder("loc_col_start").append(ctx.location[2])
                instance.builder("loc_col_end").append(ctx.location[3])

                instance.builder("caller_id").append(parent_fn_id)
                instance.builder("callee_id").append(ctx.caller_id)
                instance.builder("source_file").append(ctx.fn_context.source_file)
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

    def __init__(self, from_script_path=None):
        # TODO: support in-memory mode

        # Maps function UUIDs (_MAXRAY_TRANSFORM_ID) to FnContext instances
        self.fn_sources = {}

        self.builders = {}

        self.type_schema = {
            # Source location
            "loc_line_start": pa.int32(),
            "loc_line_end": pa.int32(),
            "loc_col_start": pa.int32(),
            "loc_col_end": pa.int32(),
            # Function calls
            "caller_id": pa.string(),
            "callee_id": pa.string(),
            "fn": pa.string(),
            "fn_call_count": pa.int32(),
            # Source info/code
            "source_file": pa.string(),
            "source": pa.string(),
            # Extracted data
            "struct_repr": pa.string(),
            "lsp_repr": pa.string(),
            "value_type": pa.string(),
            "timestamp": pa.float64(),
        }

        log_file_name = ".maxray-logs.arrow"
        if from_script_path is not None:
            self.save_to = Path(from_script_path).resolve(True).parent / log_file_name
        else:
            self.save_to = Path("/tmp") / log_file_name

        self.flush_every_records = 10_000

        self.write_context = ExitStack()

    def schema(self):
        return pa.schema([pa.field(k, v) for k, v in self.type_schema.items()])

    def builder(self, name: str):
        if name not in self.builders:
            self.builders[name] = []

        return self.builders[name]

    def flush(self):
        if not self.builders:
            logger.warning("Nothing to flush")
            return

        arrays, names = [], []
        for col_name, col_type in self.type_schema.items():
            builder = self.builders[col_name]
            arrays.append(pa.array(builder, type=col_type))
            names.append(col_name)

            builder.clear()

        batch = pa.RecordBatch.from_arrays(arrays=arrays, names=names)
        self.writer.write(batch)
        return batch

    def __enter__(self):
        CaptureLogs.instance.set(self)

        # TODO: forbid re-entry

        self.sink = self.write_context.enter_context(pa.OSFile(str(self.save_to), "wb"))
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

    @staticmethod
    def attach(*, debug: bool = False):
        """
        If
        """

        def decor(f):
            fx = xray(CaptureLogs.extractor)(f)

            @wraps(f)
            def wrapper(*args, **kwargs):
                _set_logging(debug)
                return fx(*args, **kwargs)

            return wrapper

        return decor
