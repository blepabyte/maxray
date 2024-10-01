from __future__ import annotations

from maxray.nodes import NodeContext, RayContext
from .display import Display

import ipdb
import attrs

import json
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import inspect
from functools import partial
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Callable, Generator, Iterator

import rerun as rr


class Statefool:
    """
    Session state (and utils), persisted across runs.
    """

    def __init__(self):
        self._existing_keys = {}

    def __getitem__(self, key):
        return self._existing_keys[key][1]

    def __setitem__(self, key, value):
        if key in self._existing_keys:
            v, _old_value = self._existing_keys[key]
        else:
            v = 1

        self._existing_keys[key] = (v, value)

    def define_once(self, key, factory, /, v: int = 0):
        """
        Args:
            key (Immutable + Hash + Eq): Identifies this object between steps
        """
        # Note: can use runtime value for def key for auto grouping!
        ex_o = None
        # Not thread-safe....
        if key in self._existing_keys:
            ex_v, ex_o = self._existing_keys[key]
            if ex_v >= v:
                return ex_o

        self._existing_keys[key] = (v, new_o := factory(ex_o))
        return new_o

    @property
    def display(self):
        return self.define_once("RICH_LIVE_DISPLAY", lambda _: Display())

    def enter_debugger(self, post_mortem: TracebackType | bool = False):
        with self.display.hidden():
            if post_mortem is True:
                # Needs to be an active exception
                ipdb.post_mortem()
            elif isinstance(post_mortem, TracebackType):
                ipdb.post_mortem(post_mortem)
            else:
                ipdb.set_trace()


class IterScope:
    gen: Generator
    it: Iterator

    @dataclass
    class GenMessage:
        msg: Any

    @dataclass
    class Completed:
        value: Any

    @dataclass
    class Incomplete:
        """
        Can only get a result after completion (i.e. after exiting the context)
        """

        ...

    @dataclass
    class Errored:
        """
        Exception was thrown when control was given to the iterator.
        """

        exception: Exception
        traceback: Any

    @dataclass
    class Terminated:
        """
        The iterator returned a value before consuming all generated values.
        """

        value: Any

    @dataclass
    class NotStarted: ...

    def __init__(self, rewrite):
        if not inspect.isgeneratorfunction(rewrite):
            raise ValueError("Not a generator")

        self.produce_count = 0
        self.gen = self.inverted_generator()

        self.consume_count = 0
        self.it = rewrite(self.gen)
        self.completed = False
        self.status = self.NotStarted()

    def inverted_generator(self):
        last_x = yield  # next, send
        assert isinstance(last_x, IterScope.GenMessage)

        while True:
            yield  # suspend
            next_x = None
            # Will keep iterating the same element until `it` yields (and )
            while not isinstance(next_x, IterScope.GenMessage):
                next_x = yield last_x.msg
            self.produce_count += 1
            last_x = next_x

    def push(self, x):
        self.map(x)  # discard returned value

    def map(self, x):
        if not self.completed:
            self.gen.send(self.GenMessage(x))
            try:
                x = next(self.it)
                self.consume_count += 1
                # TODO: check counters against double-iteration
            except StopIteration as si:
                self.completed = True
                self.status = self.Terminated(si.value)

            except Exception as e:
                self.completed = True
                self.status = self.Errored(e, e.__traceback__)
            # Check that the generator must have advanced
        return x

    def result(
        self,
    ) -> Completed | Incomplete | Terminated | Errored | NotStarted:
        return self.status

    @contextmanager
    def enter(self):
        self.status = self.Incomplete()
        next(self.gen)  # Start generator
        try:
            yield
        finally:
            self.gen.close()
            if not self.completed:
                try:
                    next(self.it)
                    # It yielded too many values
                    raise RuntimeError("Expected iterator to terminate")
                except StopIteration as si:
                    self.status = self.Completed(si.value)
                    self.completed = True


class RewriteContext:
    def __init__(self, name):
        self.name = name
        self._instances = ContextVar("rewrite_instance_stack")
        self._instances.set([])
        self._last_value = []

    @property
    def instances(self):
        return self._instances.get()

    @contextmanager
    def stack(self, rc: Callable[[Iterator], Iterator]):
        ih = IterScope(rc)
        self.instances.append(ih)
        try:
            with ih.enter():
                yield ih
        finally:
            self.instances.remove(ih)

            match ih.status:
                case IterScope.Errored():
                    # TODO: throw something?
                    ...

    @contextmanager
    def override(self, rc: Callable[[Iterator], Iterator]):
        ih = IterScope(rc)
        reset = self._instances.set([ih])
        try:
            with ih.enter():
                yield ih
        finally:
            self._instances.reset(reset)

    @contextmanager
    def __call__(self, rc):
        assert len(self.instances) == 0
        with self.stack(rc) as ih:
            yield ih

    def push(self, x):
        self._last_value = [x]
        for i in self.instances:
            i.push(x)

    def map(self, x):
        self._last_value = [x]
        for i in self.instances:
            x = i.map(x)
        return x

    def last(self):
        if not self._last_value:
            raise RuntimeError(f"No values pushed to {self.name} yet")
        return self._last_value[0]


class Rewriter:
    def __init__(self):
        self.by_class = {}

    def __getattr__(self, rewrite_cls_name: str) -> RewriteContext:
        if rewrite_cls_name not in self.by_class:
            self.by_class[rewrite_cls_name] = RewriteContext(rewrite_cls_name)
        return self.by_class[rewrite_cls_name]


class LoggingEncoder(json.JSONEncoder):
    def default(self, o):
        if attrs.has(type(o)):
            return attrs.asdict(o)
        return super().default(o)


class Ray(RayContext):
    """
    Captures the state of a point (syntax node) in the source code of the original program.

    One instance is created for each point in the program, that is then passed to multiple handlers.
    """

    def log(self, msg, *, level="INFO"):
        """
        Logs to Rerun with the current context if active.
        """
        match msg:
            case dict():
                try:
                    msg = json.dumps(msg, indent=2, cls=LoggingEncoder)
                except Exception:
                    msg = str(msg)
            case _ if attrs.has(type(msg)):
                msg = str(msg)

        location = Path(self.ctx.fn_context.source_file).name
        line = self.ctx.location[0]
        rr.log(f"log/{location}:{line}", rr.TextLog(msg, level=level))

    def contextmanager(self, fn: Callable[[Ray], Iterator[Any]]):
        return contextmanager(partial(fn, self))

    def scope_stack(
        self,
        iter_namespace: RewriteContext,
        iter_fn: Callable[[Ray, Any], Iterator[Any]],
    ):
        """
        Passes ourself as the first argument to a created IterScope.

        This preserves access to a view of locals() in the `with` block.
        """
        return iter_namespace.stack(partial(iter_fn, self))


R = Rewriter()

S = Statefool()
