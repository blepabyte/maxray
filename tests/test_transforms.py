from maxray import transform, xray, maxray
from maxray.transforms import NodeContext
from maxray.walkers import dbg

from contextlib import contextmanager
from dataclasses import dataclass
import functools

import pytest

from loguru import logger

logger.enable("maxray")


def increment_ints_by_one(x, ctx):
    if isinstance(x, int):
        return x + 1
    else:
        return x


def convert_ints_to_str(x, ctx):
    match x:
        case int():
            return str(x)
        case _:
            return x


def test_basic():
    @transform(increment_ints_by_one)
    def f(x):
        return x

    assert f(3) == 4


def test_type_hints():
    from typing import Any

    @transform(increment_ints_by_one)
    def f(x: Any):
        return x

    assert f(3) == 4


def test_closure_capture():
    z = 1

    @transform(convert_ints_to_str)
    def f(x):
        return x + z

    assert f(3) == "31"


def test_closure_capture_mutate():
    z = []

    @transform(increment_ints_by_one)
    def f(x):
        z.append(x)

    f(1)
    f(2)
    assert z == [2, 3]


GLOB_CONST = 5


def test_global_capture():
    @transform(convert_ints_to_str)
    def g(x):
        return x + GLOB_CONST

    assert g(3) == "35"


def test_nested_def():
    def outer():
        z = []

        @transform(increment_ints_by_one)
        def g(x):
            z.append(x)
            return x

        return g

    assert outer()(3) == 4
    assert outer()(3) == 4


def test_recursive():
    @transform(dbg)
    def countdown(n):
        if n > 0:
            return 1 + countdown(n - 1)
        else:
            return 0

    assert countdown(5) == 5


def test_fib():
    @transform(dbg)
    def fib(n: int):
        a, b = 1, 1
        for i in range(n):
            a, b = b, a + b
        return b

    fib(10)


def test_decorated():
    def outer(z):
        def wrapper(f):
            @functools.wraps(f)
            def inner(x):
                print(z)
                return f(x) + 1

            return inner

        return wrapper

    def middle(x):
        @outer("nope")
        def f(x):
            return x

        return f(x)

    @xray(dbg)
    def f(x):
        return middle(x)

    assert f(3) == 4


def test_inspect_signature():
    from inspect import signature

    sss = xray(dbg)(signature)
    print(sss(lambda x: x))


@contextmanager
def oop():
    try:
        yield 2
    finally:
        pass


def test_contextmanager():
    @maxray(lambda x, ctx: x * 2 if isinstance(x, int) else x)
    def f():
        with oop() as x:
            pass
        return x

    assert f() == 8


def test_property_access():
    @dataclass
    class A:
        x: int
        y: float

    obj = A(1, 2.3)

    @maxray(increment_ints_by_one)
    def g():
        return obj.x

    assert g() == 2


def test_method():
    class A:
        def foo(self):
            x = 1
            return str(x)

    @maxray(lambda x, ctx: (print(x), x + 1)[-1] if isinstance(x, int) else x)
    def g():
        a = A()
        return a.foo()

    assert g() == "3"


def test_recursive_self_repr():
    """
    RecursionError: maximum recursion depth exceeded while getting the str of an object

    The problem is that as part of `dbg`, uses of `self` trigger *another* `dbg` call and so on...
    """

    class X:
        def bar(self):
            pass

        def __repr__(self):
            # The problem was that self is callable
            self
            return "X"

    @xray(dbg)
    def inner():
        x = X()
        print(X.__repr__(x))
        print(f"{X()}")
        return 1

    assert inner() == 1


def test_class_method_default_arg_using_local():
    """
    Ran into this with `_PosixFlavour` in `pathlib.py` (`splitroot`, line 8)
    """

    class Flavour:
        sep = " "

        def groot(self, sep=sep):
            return sep

    @xray(dbg)
    def buggy():
        f = Flavour()
        z = f.groot()
        a = 1 + 2

    buggy()


@pytest.mark.xfail
def test_method_scope_overwrite():
    """
    So many edge cases...

    So the difference between method definitions and standard function defs is that for fn defs, the name of the function itself is made available in the calling scope (e.g. for recursion - seemingly stored as a closure within nested defs).

    For methods, we have to mangle the name so that in the exec-ed function, we can correctly reference any coexisting globally def-ed function of the same name.
    """

    def OoO():
        return 1

    class X:
        def OoO(self):
            return OoO() + 1

    @xray(dbg)
    def instance_call():
        x = X()
        return x.OoO()

    @xray(dbg)
    def static_call():
        x = X()
        return X.OoO(x)

    # method transform gives correct results, non-method errors
    assert static_call() == 2
    assert instance_call() == 2
    assert OoO() == 1


def test_static_call():
    class X:
        def f(self):
            return 1

    @xray(dbg)
    def call():
        x = X()
        return X.f(x)

    assert call() == 1


def test_private_name_mangled():
    class X:
        def __init__(self):
            self.__next()

        def __next(self):
            print("hi")

        def g(self):
            self.__next()

    @xray(dbg)
    def bad():
        x = X()
        x.g()

    bad()


def test_static_name_mangled():
    # pynvml/smi.py
    class smi:
        __instance = None

        def getInstance(self):
            return self.__instance

        def getInstanceNamed(self):
            return smi.__instance

        @staticmethod
        def getInstanceStatic():
            return smi.__instance

    @xray(dbg)
    def useInstance():
        s = smi()
        assert s.getInstance() is None
        assert s.getInstanceNamed() is None

        # these 2 fail
        assert s.getInstanceStatic() is None
        assert smi.getInstanceStatic() is None

    useInstance()


def test_super():
    class A:
        def to(self):
            return 1

    class B(A):
        def to(self):
            return super().to() + 1

    @maxray(dbg)
    def oop():
        b = B()
        return b.to()

    assert oop() == 2


def test_closure_reuse():
    x = []

    @maxray(dbg)
    def foo():
        # nonlocal x
        x.append(1)

    @maxray(dbg)
    def bar():
        x.append(2)

    foo()
    bar()
    assert len(x) == 2


def test_double_decorators_with_locals():
    x = []

    @maxray(dbg)
    @maxray(dbg)
    def foo():
        # nonlocal x
        x.append(1)

    foo()


def test_xray_immutable():
    @maxray(lambda x, ctx: x * 10 if isinstance(x, float) else x)
    @xray(increment_ints_by_one)
    def foo():
        x = 1
        y = 2.0
        return x, y

    assert foo() == (1, 200.0)


def test_walk_callable_side_effects():
    counter = 0

    def slide(x, ctx):
        nonlocal counter
        if callable(x):
            counter += 1

    @xray(slide)
    def foo():
        f = lambda x: x
        for i in range(10):
            f(i)

    foo()

    assert counter == 11


def test_match():
    @maxray(lambda x, ctx: x * 2 if isinstance(x, str) else x)
    def matcher(x):
        match x:
            case int():
                return str(x)
            case str() as y:
                return y
            case {"aaa": {"bbb": ccc}}:
                return ccc

    assert matcher(1) == "11"
    assert matcher("foo") == "foofoofoofoo"


def test_multi_decorators():
    decor_count = []

    def dec(f):
        decor_count.append(1)
        return f

    # External decorator applied first: is executed (side effects) but is ignored/wiped and does not affect the generated function at all
    @maxray(increment_ints_by_one)
    @dec
    def f(x):
        return x

    assert f(2) == 3
    assert len(decor_count) == 1

    # Works properly when applied last: is wiped for the transform, but is subsequently applied properly to the transformed function
    @dec
    @maxray(lambda x, ctx: x - 1 if isinstance(x, int) else x)
    def f(x):
        return x

    assert f(2) == 1
    assert len(decor_count) == 2


@pytest.mark.xfail
def test_unhashable_callable():
    class X:
        def __call__(self):
            return 1

    X.__hash__ = None

    @maxray(increment_ints_by_one)
    def uh():
        z = X()
        return z()

    assert uh() == 3


def test_junk_annotations():
    @maxray(convert_ints_to_str)
    def outer():
        def inner(x: ASDF = 0, *, y: SDFSDF = 100) -> AAAAAAAAAAA:
            return x + y

        return inner(2)

    assert outer() == "2100"


def test_call_counts():
    calls = []

    def track_call_counts(x, ctx: NodeContext):
        calls.append(ctx.fn_context.call_count.get())
        return x

    @xray(track_call_counts)
    def f(x):
        return x

    f(1)
    f(1)
    assert set(calls) == {1, 2}

    f(1)
    assert set(calls) == {1, 2, 3}


def test_call_counts_recursive():
    calls = []

    def track_call_counts(x, ctx: NodeContext):
        if ctx.id in ["name/f", "call/f(x - 1)"]:
            calls.append(ctx.fn_context.call_count.get())
        return x

    @xray(track_call_counts)
    @xray(dbg)
    def f(x):
        if x > 0:
            return f(x - 1)
        return 1

    f(3)
    assert calls == [1, 2, 3, 3, 2, 1]


def test_empty_return():
    @xray(dbg)
    def empty_returns():
        return

    assert empty_returns() is None


def test_scope_passed():
    found_scope = {}

    def get_scope(x, ctx):
        nonlocal found_scope
        found_scope.update(ctx.local_scope)
        return x

    @xray(get_scope, pass_scope=True)
    def f(n):
        z = 3
        return n

    assert f(1) == 1

    assert "z" in found_scope
    assert found_scope["z"] == 3


def test_class_super():
    class F:
        def __init__(self, **kwargs):
            super().__init__()

        def f(self):
            return 1

    class G(F):
        def __init__(self, x, y):
            self.x = x
            self.y = y
            super().__init__(x=x)

    @maxray(increment_ints_by_one)
    def fn(f, g):
        # this errors
        G(1, 2)
        G(1, 2)
        return 4

    @maxray(increment_ints_by_one)
    def fn_works(f, g):
        # this doesn't
        F
        G
        G(1, 2)
        G(1, 2)
        return 4

    # When given an *instance*, the __init__ in F overwrites G.__init__
    # However, given just F, it correctly patches F.__init__

    f_instance = F()
    g_instance = G(1, 2)
    assert fn(f_instance, g_instance) == fn_works(f_instance, g_instance) == 5


@pytest.mark.xfail
def test_class_super_explicit():
    class H0:
        def __init__(self, **kwargs):
            super().__init__()

        def f(self):
            return 1

    class H1(H0):
        def __init__(self, x, y):
            self.x = x
            self.y = y
            # This is currently not handled correctly
            super(H1, self).__init__(x=x)

    @maxray(increment_ints_by_one)
    def fn():
        H1(1, 2)
        H1(1, 2)
        return 4

    assert fn() == 5


def test_super_classmethod():
    class S0:
        def __init__(self, **kwargs):
            super().__init__()

        @classmethod
        def foo(cls):
            return 1

    class S1(S0):
        @classmethod
        def foo(cls):
            return super().foo()

    @maxray(increment_ints_by_one)
    def fff():
        return S1.foo()

    assert fff() == 4


def test_partialmethod():
    from functools import partialmethod

    # TQDM threw an error from partialmethod via @env_wrap but can't seem to reproduce
    @xray(dbg)
    def run_part():
        class X:
            def set_state(self, active: bool):
                self.active = active

            set_active = partialmethod(set_state, True)

        x = X()
        x.set_active()

        assert x.active

    run_part()


def test_caller_id():
    f1_id = None
    f2_id = None

    def collect_ids(x, ctx: NodeContext):
        if ctx.source == "f1()":
            nonlocal f1_id
            f1_id = ctx.caller_id
        elif ctx.source == "f2()":
            nonlocal f2_id
            f2_id = ctx.caller_id

    def f1():
        return 1

    def f2():
        return 2

    @xray(collect_ids)
    def func():
        f1()
        f2()

    func()

    assert f1._MAXRAY_TRANSFORM_ID == f1_id
    assert f2._MAXRAY_TRANSFORM_ID == f2_id
    assert f1_id != f2_id


global_x = 0


def test_global_update():
    def mutate_global():
        global global_x
        global_x = 37

    @xray(dbg)
    def global_ops():
        if global_x == 0:
            mutate_global()
        return global_x

    assert global_ops() == 37
    assert global_x == 37


def test_global_set():
    def set_global():
        global global_y
        global_y = 101

    @xray(dbg)
    def make_global():
        set_global()

    make_global()
    assert global_y == 101


def isna(cls):
    return 2


class Framed:
    def isna(self):
        # should call isna in module scope rather than recursing forever into this call
        return isna(self)


def test_qualified_invoke():
    @xray(dbg)
    def check_isna():
        f = Framed()
        return Framed.isna(f)

    assert check_isna() == 2


def test_qualified_init():
    class A:
        def __init__(self):
            self.a_prop = 101

    class B(A):
        def __init__(self):
            A.__init__(self)

    @xray(dbg)
    def get_prop():
        return B().a_prop

    assert get_prop() == 101
