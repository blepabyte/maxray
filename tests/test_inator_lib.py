from maxray.inators.core import IterScope, persist_type

from dataclasses import dataclass

import pytest


def normal_it(gen):
    for x in gen:
        yield x + 1
    return "ok?"


def test_cross_iter():
    with (ih := IterScope(normal_it)).enter():
        ih.push(1)
        ih.push(2)
        ih.push(3)

    assert ih.result() == IterScope.Completed("ok?")


def test_incomplete_iter():
    with (ih := IterScope(normal_it)).enter():
        assert ih.result() == IterScope.Incomplete()


def test_early_termination_iter():
    def terminate_it(gen):
        for x in gen:
            yield x + 1
            break
        return "termd"

    with (ih := IterScope(terminate_it)).enter():
        ih.push(1)
        ih.push(2)
        ih.push(3)

    assert ih.result() == IterScope.Terminated("termd")


def test_exception_iter():
    def exception_it(gen):
        for x in gen:
            yield x + 1
            raise RuntimeError("oh no!")

    with (ih := IterScope(exception_it)).enter():
        ih.push(1)
        ih.push(2)
        ih.push(3)

    result = ih.result()
    assert isinstance(result, IterScope.Errored)
    assert isinstance(result.exception, RuntimeError)


def test_double_yield_iter():
    def double_yield_it(gen):
        for x in gen:
            yield x + 1
            yield x + 1

    with pytest.raises(RuntimeError):
        with (ih := IterScope(double_yield_it)).enter():
            ih.push(1)
            ih.push(2)
            ih.push(3)


# this loops forever...
# def test_double_consume_iter():
#     def double_consume_it(gen):
#         yield from []
#         for x in gen:
#             continue

#     with pytest.raises(RuntimeError):
#         with (ih := IterHandle(double_consume_it)).enter():
#             ih.push(1)
#             ih.push(2)
#             ih.push(3)


def test_not_a_generator():
    def forgot_to_yield(gen):
        for x in gen:
            continue

    with pytest.raises(ValueError):
        with (ih := IterScope(forgot_to_yield)).enter():
            ih.push(1)


def test_interrupted():
    def robust_it(gen):
        processed = 0
        try:
            for x in gen:
                processed += 1
                yield x
        finally:
            return {"processed": processed}

    with pytest.raises(RuntimeError):
        with (ih := IterScope(robust_it)).enter():
            ih.push(1)
            ih.push(2)
            raise RuntimeError()
            ih.push(3)

    match ih.result():
        case IterScope.Completed({"processed": 2}):
            assert True
        case _:
            assert False, ih.result()


def test_internal_interrupted():
    def robust_it(gen):
        processed = 0
        try:
            for x in gen:
                processed += 1
                yield x
                raise ValueError()
        finally:
            return {"processed": processed}

    with pytest.raises(RuntimeError):
        with (ih := IterScope(robust_it)).enter():
            ih.push(1)
            ih.push(2)
            raise RuntimeError()
            ih.push(3)

    match ih.result():
        case IterScope.Terminated({"processed": 1}):
            assert True
        case _:
            assert False, ih.result()


def test_type_identity_on_reload():
    @persist_type
    @dataclass(frozen=True)
    class Foo:
        x: int

        def oof(self):
            return 1 + self.x

    qn1 = Foo.__qualname__
    i1 = Foo(2)
    assert i1.oof() == 3

    @persist_type
    @dataclass(frozen=True)
    class Foo:
        x: int

        def oof(self):
            return 2 + self.x

    qn2 = Foo.__qualname__
    i2 = Foo(2)

    assert qn1 == qn2
    assert i1 == i2
    assert type(i1) is type(i2)
    assert i1.oof() == i2.oof() == 4
    assert i1 in {i2}
    assert i2 in {i1}
