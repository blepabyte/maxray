from .transforms import recompile_fn_with_transform, NodeContext
from .function_store import FunctionStore, set_property_on_functionlike

import inspect
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps, _lru_cache_wrapper
from typing import Any, Callable, Optional
from result import Result, Ok, Err

import os

from loguru import logger

if not os.environ.get("MAXRAY_LOG_LEVEL"):
    # Avoid logspam for users of the library
    logger.disable("maxray")


def _set_logging(enabled: bool):
    if enabled:
        logger.enable("maxray")
    else:
        logger.disable("maxray")


@contextmanager
def _with_logging():
    try:
        _set_logging(True)
        yield
    finally:
        _set_logging(False)


def transform(writer):
    """
    Decorator that rewrites the source code of a function to wrap every single expression by passing it through the `writer(expr, ctx)` callable.
    """

    def inner(fn):
        match recompile_fn_with_transform(fn, writer, is_maxray_root=True):
            case Ok(trans_fn):
                return wraps(fn)(trans_fn)
            case Err(err):
                logger.error(err)
                return fn

    return inner


def xray(walker, **kwargs):
    """
    Immutable version of `maxray` - expressions are passed to `walker` but its return value is ignored and the original code execution is left unchanged.
    """
    return maxray(walker, **kwargs, mutable=False)


_GLOBAL_SKIP_MODULES = {
    "builtins",  # duh... object.__init__ is a wrapper_descriptor
    "abc",  # excessive inheritance and super calls in scikit-learn
    "inspect",  # don't want to screw this module up
    "pathlib",  # internally used in transform for checking source file exists
    "re",  # internals of regexp have a lot of uninteresting step methods
    "copy",  # pytorch spends so much time in here
    "functools",  # partialmethod causes mis-bound `self` in TQDM
    "asyncio",  # AttributeError: 'NoneType' object has no attribute 'new_event_loop'
    "logging",  # global _loggerClass probably causes problems
    "typing",
    "ast",
    "importlib",
    "ctypes",
    "loguru",  # used internally in transform - accidental patching will cause inf recursion
    "uuid",  # used for generating _MAXRAY_TRANSFORM_ID
    "urllib3",
    "aiohttp",  # something icky about web.RouteTableDef() when route decorated from submodule
    "maxray",
    "git",
}
# TODO: probably just skip the entire Python standard library...


@dataclass
class W_erHook:
    impl_fn: Callable
    active_call_state: ContextVar[bool]
    writer_active_call_state: ContextVar[bool]

    descend_predicate: Callable
    "Determines whether to descend into the source code of callables in the given function context."

    mutable: bool

    # each walker defines names to skip and we skip recursive transform if *any* walker asks to skip


_MAXRAY_REGISTERED_HOOKS: list[W_erHook] = []


def descend_allowed(x, ctx: NodeContext):
    num_active_hooks = 0
    for hook in _MAXRAY_REGISTERED_HOOKS:
        if hook.active_call_state.get():
            num_active_hooks += 1
            if not hook.descend_predicate(x, ctx):
                return False
    return num_active_hooks > 0


_GLOBAL_WRITER_ACTIVE_FLAG = ContextVar("writer_active (global)", default=False)

# We don't want to recompile the same function over and over - so our cache needs to be global
# TODO: add cache by source code location?
_MAXRAY_FN_CACHE = dict()

wrapper_descriptor = type(object.__init__)
method_wrapper = type(type.__call__.__get__(type))


def transform_precheck(x, ctx: NodeContext):
    if not callable(x):
        return False

    if isinstance(x, type):  # Not a function
        # Also, we want __getattribute__ to be the bound method for an instance
        return False

    if isinstance(x, (wrapper_descriptor, method_wrapper)):
        return False

    try:
        # Avoid calling getattr since things like DataFrames override it (and cause recursion errors)
        x.__getattribute__("_MAXRAY_NOTRANSFORM")
        return False
    except AttributeError:
        pass

    try:
        x.__getattribute__("_MAXRAY_TRANSFORM_ID")
        return False
    except AttributeError:
        pass

    # __module__: The name of the module the function was defined in, or None if unavailable.
    if getattr(x, "__module__", None) in _GLOBAL_SKIP_MODULES:
        return False
    # TODO: also check ctx.fn_context.module?

    return True


def callable_allowed_for_transform(x, ctx: NodeContext):
    if not transform_precheck(x, ctx):
        return False

    module_path = ctx.fn_context.module.split(".")
    if module_path[0] in _GLOBAL_SKIP_MODULES:
        return False
    # TODO: deal with nonhashable objects and callables and other exotic types properly
    return (
        callable(x)  # super() has getset_descriptor instead of proper __dict__
        and hasattr(x, "__dict__")
        # and "_MAXRAY_TRANSFORMED" not in x.__dict__
        # since we no longer rely on caching, we don't need to check for __hash__
        # and callable(getattr(x, "__hash__", None))
        and getattr(type(x), "__module__", None) not in {"ctypes"}
        and (
            inspect.isfunction(x)
            or inspect.ismethod(x)
            or isinstance(x, _lru_cache_wrapper)
        )
    )


def instance_allowed_for_transform(x, ctx: NodeContext):
    """
    Checks if x is a type with dunder methods can be correctly transformed.
    """
    if type(x) is not type:
        # Filter out weird stuff that would get through with an isinstance check
        # e.g. loguru: _GeneratorContextManager object is not an iterator (might be a separate bug)
        return False

    if getattr(x, "__module__", None) in _GLOBAL_SKIP_MODULES:
        return False

    return True


def _inator_inator(restrict_modules: Optional[list[str]] = None):
    """
    Control configuration to determine which, and how, callables get recursively transformed.
    """

    def callable_precheck(x, ctx: NodeContext):
        """
        Is x a "normal" pure-Python function that we can safely substitute with its transformed version?
        """
        raise NotImplementedError()

    def instance_precheck(x, ctx: NodeContext):
        raise NotImplementedError()

    def _maxray_walker_handler(x, ctx: NodeContext, autotransform=True):
        # 1.  logic to recursively patch callables
        # 1a. special-case callables: __init__ and __call__
        if not autotransform:
            # Disables recursively attaching handler to all nested calls
            pass

        x_def_module = getattr(x, "__module__", "")
        if x_def_module is None:
            x_def_module = ""
        if restrict_modules is not None and not any(
            x_def_module.startswith(mod) for mod in restrict_modules
        ):
            pass

        elif instance_allowed_for_transform(x, ctx):
            # TODO: Delay non-init methods until we actually observe an instance?
            for dunder in ["__init__", "__call__"]:
                try:
                    dunder_fn = getattr(x, dunder)
                    if transform_precheck(dunder_fn, ctx):
                        match recompile_fn_with_transform(
                            dunder_fn,
                            _maxray_walker_handler,
                            special_use_instance_type=x,
                            triggered_by_node=ctx,
                        ):
                            case Ok(init_patch):
                                logger.debug(f"Patching {dunder} for class {x}")
                                setattr(x, dunder, init_patch)
                            case Err(_err):
                                # TODO: test that errors are reported somewhere
                                set_property_on_functionlike(
                                    dunder_fn, "_MAXRAY_NOTRANSFORM", True
                                )
                except AttributeError:
                    pass

        # 1b. normal functions or bound methods or method descriptors like @classmethod and @staticmethod
        elif callable_allowed_for_transform(x, ctx):
            # We can only cache functions - as caching invokes __hash__, which may fail badly on incompletely-initialised class instances w/ __call__ methods, like torch._ops.OpOverload
            if x in _MAXRAY_FN_CACHE:
                x = _MAXRAY_FN_CACHE[x]
            elif not descend_allowed(x, ctx):
                # TODO: deprecate
                # user-defined filters for which nodes (not) to descend into
                pass
            else:
                # TODO: fixup control flow
                match recompile_fn_with_transform(
                    x, _maxray_walker_handler, triggered_by_node=ctx
                ):
                    case Ok(x_trans):
                        # NOTE: x_trans now has _MAXRAY_TRANSFORMED field to True
                        with_fn = FunctionStore.get(x_trans._MAXRAY_TRANSFORM_ID)

                        # This does not apply when accessing X.method - only X().method
                        if inspect.ismethod(x):
                            # if with_fn.method is not None:
                            # Two cases: descriptor vs bound method
                            match x.__self__:
                                case type():
                                    # Descriptor
                                    logger.debug(
                                        f"monkey-patching descriptor method {x.__name__} on type {x.__self__}"
                                    )
                                    parent_cls = x.__self__
                                case _:
                                    # Bound method
                                    logger.debug(
                                        f"monkey-patching bound method {x.__name__} on type {type(x.__self__)}"
                                    )
                                    parent_cls = type(x.__self__)

                            self_cls = parent_cls
                            if with_fn.data.method_info.defined_on_cls is not None:
                                parent_cls = with_fn.data.method_info.defined_on_cls

                            # Monkey-patching the methods. Probably unsafe and unsound
                            # Descriptor guide: https://docs.python.org/3/howto/descriptor.html

                            # Sanity check: check that our patch target is identical to the unbound version of the method (to prevent patching on the wrong class)
                            supposed_x = getattr(parent_cls, x.__name__, None)
                            if hasattr(supposed_x, "__func__"):
                                supposed_x = supposed_x.__func__
                            if supposed_x is x.__func__ and supposed_x is not None:
                                setattr(parent_cls, x.__name__, x_trans)
                                x_patched = x_trans.__get__(x.__self__, self_cls)
                            else:
                                # Because any function can be assigned as a member of the class with an arbitrary name...
                                logger.warning(
                                    "Could not monkey-patch because instance is incorrect"
                                )
                                set_property_on_functionlike(
                                    x, "_MAXRAY_NOTRANSFORM", True
                                )
                                x_patched = x

                            # We don't bother caching methods as they're monkey-patched
                            # SOUNDNESS: a package might manually keep references to __init__ around to later call them - but we'd just end up recompiling those as well
                        else:
                            x_patched = x_trans
                            _MAXRAY_FN_CACHE[x] = x_patched
                        x = x_patched

                    case Err(e):
                        # Speedup by not trying to recompile (getsource involves filesystem lookup) the same bad function over and over
                        set_property_on_functionlike(x, "_MAXRAY_NOTRANSFORM", True)
                        logger.warning(
                            f"Failed to transform in walker handler: {e} {x.__qualname__}"
                        )

        # We ignore writer calls triggered by code execution in other writers to prevent easily getting stuck in recursive hell
        # This happens *after* checking and patching callables to still allow for explicitly patching a callable/method by calling this handler
        if _GLOBAL_WRITER_ACTIVE_FLAG.get():
            return x

        # 2. run the active hooks
        global_write_active_token = _GLOBAL_WRITER_ACTIVE_FLAG.set(True)
        try:
            for walk_hook in _MAXRAY_REGISTERED_HOOKS:
                # Our recompiled fn sets and unsets a contextvar whenever it is active
                if not walk_hook.active_call_state.get():
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

    return _maxray_walker_handler


def maxray(
    writer: Callable[[Any, NodeContext], Any],
    skip_modules=frozenset(),
    *,
    root_inator=_inator_inator(),
    mutable=True,
    pass_scope=False,
    initial_scope={},
    assume_transformed=False,
):
    """
    A transform that recursively hooks into all further calls made within the function, so that `writer` will (in theory) observe every single expression evaluated by the Python interpreter occurring as part of the decorated function call.

    There are some limitations to be aware of:
    - Be careful to avoid infinite recursion: the source code of the writer will not be transformed but it may call methods that have been monkey-patched that result in more calls to the writer function.
    - Objects that are not yet fully initialised may not behave as expected - e.g. repr may throw an error because of a missing property
    """

    ACTIVE_FLAG = ContextVar(f"maxray_active for <{writer}>", default=False)
    WRITER_ACTIVE_FLAG = ContextVar(f"writer_active for <{writer}>", default=False)

    # Resolves decorators in the local scope that aren't "closure"-d over
    # frame = inspect.currentframe()
    # try:
    #     caller_locals = frame.f_back.f_locals
    # except Exception as e:
    #     logger.exception(e)
    #     logger.error("Couldn't get locals")
    #     caller_locals = {}
    # finally:
    #     del frame

    # TODO: allow configuring injection of variables into exec scope
    caller_locals = initial_scope

    def recursive_transform(fn):
        _MAXRAY_REGISTERED_HOOKS.append(
            W_erHook(
                writer,
                ACTIVE_FLAG,
                WRITER_ACTIVE_FLAG,
                lambda x, ctx: ctx.fn_context.module.split(".")[0] not in skip_modules,
                mutable=mutable,
            )
        )

        # Fixes `test_double_decorators_with_locals`: repeated transforms are broken because stuffing closures into locals doesn't work the second time around
        if hasattr(fn, "_MAXRAY_TRANSFORMED") or assume_transformed:
            fn_transform = fn
        else:
            match recompile_fn_with_transform(
                fn,
                root_inator,
                override_scope=caller_locals,
                pass_scope=pass_scope,
                is_maxray_root=True,
            ):
                case Ok(fn_transform):
                    pass
                case Err(err):
                    # Do not allow silently failing if a function has been explicitly annotated with @xray or the like
                    raise RuntimeError(f"{err}: Failed to transform {fn}")

        # BUG: We can't do @wraps if it's a callable instance, right?
        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def fn_with_context_update(*args, **kwargs):
                prev_token = ACTIVE_FLAG.set(True)
                try:
                    return await fn_transform(*args, **kwargs)
                finally:
                    ACTIVE_FLAG.reset(prev_token)
        else:

            @wraps(fn)
            def fn_with_context_update(*args, **kwargs):
                prev_token = ACTIVE_FLAG.set(True)
                try:
                    return fn_transform(*args, **kwargs)
                finally:
                    ACTIVE_FLAG.reset(prev_token)

        fn_with_context_update._MAXRAY_TRANSFORMED = True
        # TODO: set correctly everywhere
        # fn_with_context_update._MAXRAY_TRANSFORM_ID = ...
        return fn_with_context_update

    return recursive_transform
