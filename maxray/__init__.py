from .transforms import recompile_fn_with_transform, NodeContext

import inspect
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable
from result import Result, Ok, Err

from loguru import logger


def transform(writer):
    """
    Decorator that rewrites the source code of a function to wrap every single expression by passing it through the `writer(expr, ctx)` callable.
    """

    def inner(fn):
        match recompile_fn_with_transform(fn, writer):
            case Ok(trans_fn):
                return wraps(fn)(trans_fn)
            case Err(err):
                logger.error(err)
                return fn

    return inner


def xray(walker):
    """
    Immutable version of `maxray` - expressions are passed to `walker` but its return value is ignored and the original code execution is left unchanged.
    """
    return maxray(walker, mutable=False)


_GLOBAL_SKIP_MODULES = {
    "abc",  # excessive inheritance and super calls in scikit-learn
    "inspect",  # don't want to screw this module up
    "pathlib",  # internally used in transform for checking source file exists
    "re",  # internals of regexp have a lot of uninteresting step methods
    "copy",  # pytorch spends so much time in here
}


@dataclass
class W_erHook:
    impl_fn: Callable
    active_call_state: ContextVar[bool]
    writer_active_call_state: ContextVar[bool]
    skip_modules: set
    only_modules: set
    mutable: bool

    # each walker defines names to skip and we skip recursive transform if *any* walker asks to skip


_MAXRAY_REGISTERED_HOOKS: list[W_erHook] = []

_GLOBAL_WRITER_ACTIVE_FLAG = ContextVar("writer_active (global)", default=False)

# We don't want to recompile the same function over and over - so our cache needs to be global
_MAXRAY_FN_CACHE = dict()


def callable_allowed_for_transform(x, ctx: NodeContext):
    module_path = ctx.fn_context.module.split(".")
    if module_path[0] in _GLOBAL_SKIP_MODULES:
        return False
    return not hasattr(x, "_MAXRAY_TRANSFORMED") and callable(x)


def _maxray_walker_handler(x, ctx):
    # We ignore writer calls triggered by code execution in other writers to prevent easily getting stuck in recursive hell
    if _GLOBAL_WRITER_ACTIVE_FLAG.get():
        return x

    # 1. logic to recursively patch callables
    if callable_allowed_for_transform(x, ctx):
        if x in _MAXRAY_FN_CACHE:
            return _MAXRAY_FN_CACHE[x]

        # Our recompiled fn sets and unsets a contextvar whenever it is active
        match recompile_fn_with_transform(x, _maxray_walker_handler):
            case Ok(x_trans):
                # NOTE: x_trans now has _MAXRAY_TRANSFORMED field to True
                if inspect.ismethod(x):
                    # Two cases: descriptor vs bound method
                    # TODO: handle callables and .__call__ patching
                    match x.__self__:
                        case type():
                            # Descriptor
                            logger.warning(
                                f"monkey-patching descriptor method {x.__name__} on type {x.__self__}"
                            )
                            parent_cls = x.__self__
                        case _:
                            # Bound method
                            logger.warning(
                                f"monkey-patching bound method {x.__name__} on type {type(x.__self__)}"
                            )
                            parent_cls = type(x.__self__)

                    # Monkey-patching the methods. Probably unsafe and unsound
                    setattr(parent_cls, x.__name__, x_trans)
                    x_patched = getattr(
                        x.__self__, x.__name__
                    )  # getattr turns class descriptors (@classmethod) into bound methods

                    # We don't bother caching methods as they're monkey-patched
                    # SOUNDNESS: a package might manually keep references to __init__ around to later call them - but we'd just end up recompiling those as well
                else:
                    x_patched = x_trans
                    _MAXRAY_FN_CACHE[x] = x_patched
                x = x_patched

            case Err(e):
                # Cache failures
                _MAXRAY_FN_CACHE[x] = x
                # Errors in functions that have been recursively compiled are unimportant
                logger.trace(f"Failed to transform in walker handler: {e}")

    # 2. run the active hooks
    global_write_active_token = _GLOBAL_WRITER_ACTIVE_FLAG.set(True)
    try:
        for walk_hook in _MAXRAY_REGISTERED_HOOKS:
            if not walk_hook.active_call_state.get():
                continue

            if ctx.fn_context.module in walk_hook.skip_modules:
                continue

            # Set the writer active flag
            write_active_token = walk_hook.writer_active_call_state.set(True)
            if walk_hook.mutable:
                x = walk_hook.impl_fn(x, ctx)
            else:
                walk_hook.impl_fn(x, ctx)
            walk_hook.writer_active_call_state.reset(write_active_token)
    finally:
        _GLOBAL_WRITER_ACTIVE_FLAG.reset(global_write_active_token)

    return x


def maxray(
    writer: Callable[[Any, NodeContext], Any], skip_modules=frozenset(), *, mutable=True
):
    """
    A transform that recursively hooks into all further calls made within the function, so that `writer` will (in theory) observe every single expression evaluated by the Python interpreter occurring as part of the decorated function call.

    There are some limitations to be aware of:
    - Be careful to avoid infinite recursion: the source code of the writer will not be transformed but it may call methods that have been monkey-patched that result in more calls to the writer function.
    - Objects that are not yet fully initialised may not behave as expected - e.g. repr may throw an error because of a missing property
    """

    ACTIVE_FLAG = ContextVar(f"maxray_active for <{writer}>", default=False)
    WRITER_ACTIVE_FLAG = ContextVar(f"writer_active for <{writer}>", default=False)

    def recursive_transform(fn):
        _MAXRAY_REGISTERED_HOOKS.append(
            W_erHook(
                writer,
                ACTIVE_FLAG,
                WRITER_ACTIVE_FLAG,
                skip_modules,
                set(),
                mutable=mutable,
            )
        )

        # Fixes `test_double_decorators_with_locals`: repeated transforms are broken because stuffing closures into locals doesn't work the second time around
        if hasattr(fn, "_MAXRAY_TRANSFORMED"):
            fn_transform = fn
        else:
            match recompile_fn_with_transform(fn, _maxray_walker_handler):
                case Ok(fn_transform):
                    pass
                case Err(err):
                    # Errors are only displayed at top-level, when the user has manually annotated a function with @xray or the like
                    logger.error(err)
                    return fn

        # BUG: We can't do @wraps if it's a callable instance, right?
        @wraps(fn)
        def fn_with_context_update(*args, **kwargs):
            # already active on stack
            if ACTIVE_FLAG.get():
                return fn_transform(*args, **kwargs)

            ACTIVE_FLAG.set(True)
            try:
                return fn_transform(*args, **kwargs)
            finally:
                ACTIVE_FLAG.set(False)

        fn_with_context_update._MAXRAY_TRANSFORMED = True
        return fn_with_context_update

    return recursive_transform
