from maxray.inators.core import IterScope
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
