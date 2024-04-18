from maxray import transform, xray, maxray
from maxray.walkers import dbg

from contextlib import contextmanager
from dataclasses import dataclass
import functools


def test_basic():
    @transform(lambda x, ctx: x + 1 if isinstance(x, int) else x)
    def f(x):
        return x

    assert f(3) == 4


def test_type_hints():
    from typing import Any

    @transform(lambda x, ctx: x + 1 if isinstance(x, int) else x)
    def f(x: Any):
        return x

    assert f(3) == 4


def test_closure_capture():
    z = 1

    @transform(lambda x, ctx: x + 1 if isinstance(x, int) else x)
    def f(x):
        return x + z

    assert f(3) == 6


def test_closure_capture_mutate():
    z = []

    @transform(lambda x, ctx: x + 1 if isinstance(x, int) else x)
    def f(x):
        z.append(x)

    f(1)
    f(2)
    assert z == [2, 3]


GLOB_CONST = 5


def test_global_capture():
    @transform(lambda x, ctx: x + 1 if isinstance(x, int) else x)
    def g(x):
        return x + GLOB_CONST

    assert g(3) == 10


def test_nested_def():
    def outer():
        z = []

        @transform(lambda x, ctx: x + 1 if isinstance(x, int) else x)
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
        yield "foo"
    finally:
        pass


def test_contextmanager():
    @xray(dbg)
    def f():
        with oop() as x:
            pass
        return x

    assert f() == "foo"


def test_property_access():
    @dataclass
    class A:
        x: int
        y: float

    obj = A(1, 2.3)

    @maxray(lambda x, ctx: x + 1 if isinstance(x, int) else x)
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

    assert g() == "2"


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
    def bad():
        x = X()
        return x.OoO()

    assert bad() == 2
    assert OoO() == 1


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
    @xray(lambda x, ctx: x + 1 if isinstance(x, int) else x)
    def foo():
        # Currently assumes that literals/constants are not wrapped (they're uninteresting anyways)
        x = 1
        y = 2.0
        return x, y

    assert foo() == (1, 20.0)


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
