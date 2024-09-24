from .transforms import recompile_fn_with_transform, NodeContext
from .function_store import FunctionStore, set_property_on_functionlike

import inspect
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps, _lru_cache_wrapper
from typing import Any, Callable, Optional, TypeVar
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
    # Don't patch ourself!
    "maxray",
    # Nor core parts of the language or modules we use internally
    "builtins",
    "ctypes",
    "importlib",
    "inspect",
    "typing",
    "ast",
    "pathlib",
    "uuid",  # used for generating _MAXRAY_TRANSFORM_ID
    # Uninteresting standard library modules
    # TODO: probably just skip most of the entire Python standard library...
    "abc",  # excessive inheritance and super calls in scikit-learn
    "re",  # internals of regexp have a lot of uninteresting step methods
    "copy",  # pytorch spends so much time in here
    "collections",
    # Libraries that are too weird to correctly patch
    "pytest",
    # Libraries possibly not fully working yet
    "logging",  # global _loggerClass probably causes problems
    "functools",  # partialmethod causes mis-bound `self` in TQDM
    "loguru",  # used internally in transform - accidental patching will cause inf recursion
    "urllib3",
    "aiohttp",  # something icky about web.RouteTableDef() when route decorated from submodule
    "unittest",
}


@dataclass(frozen=True)
class TransformSettings:
    forbid_modules: frozenset[str] = frozenset(_GLOBAL_SKIP_MODULES)
    """
    Functions and classes belonging to these modules are never transformed.

    Takes precdence over `restrict_modules`.
    """

    restrict_modules: frozenset[str] | None = None
    """
    If `None`, has no effect. Otherwise, *only* functions and classes belonging to these modules are transformed.
    """

    pass_local_scopes: bool = False
    """
    Whether to always populate `NodeContext.local_scope` with the evaluation of `locals()` at every node.
    """

    def update(self, forbid_modules, restrict_modules, pass_local_scopes):
        def or_empty_set(s):
            return frozenset([] if s is None else s)

        if self.restrict_modules is not None or restrict_modules is not None:
            restrict_modules = or_empty_set(self.restrict_modules).union(
                or_empty_set(restrict_modules)
            )

        return TransformSettings(
            forbid_modules=self.forbid_modules.union(forbid_modules),
            restrict_modules=restrict_modules,
            pass_local_scopes=self.pass_local_scopes or pass_local_scopes,
        )


# Making this a contextvar could allow disabling transforms on non-main threads, which might fix a few bugs?
_MAXRAY_TRANSFORM_SETTINGS = TransformSettings()


@dataclass
class W_erHook:
    impl_fn: Callable
    active_call_state: ContextVar[bool]
    writer_active_call_state: ContextVar[bool]
    mutable: bool

    # each walker defines names to skip and we skip recursive transform if *any* walker asks to skip


_MAXRAY_REGISTERED_HOOKS: list[W_erHook] = []

_GLOBAL_WRITER_ACTIVE_FLAG = ContextVar("writer_active (global)", default=False)

# We don't want to recompile the same function over and over - so our cache needs to be global
# TODO: add cache by source code location?
_MAXRAY_FN_CACHE = dict()

_MAXRAY_NOCOMPILE = False

wrapper_descriptor = type(object.__init__)
method_wrapper = type(type.__call__.__get__(type))


@contextmanager
def nocompile():
    global _MAXRAY_NOCOMPILE
    _MAXRAY_NOCOMPILE = True
    try:
        yield
    finally:
        _MAXRAY_NOCOMPILE = False


def full_module(obj):
    qual_module = getattr(obj, "__module__", None)
    if qual_module is None:
        return ""
    return qual_module


def base_module(obj):
    return full_module(obj).split(".")[0]


def transform_precheck(x, ctx: NodeContext):
    if not callable(x):
        return False

    if isinstance(x, type):  # Not a function
        # Also, we want __getattribute__ to be the bound method for an instance
        return False

    if type(x).__module__ == "types":
        # types.GenericAlias behaves like a type despite not being an instance of type...
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
    if base_module(x) in _GLOBAL_SKIP_MODULES:
        return False

    # dot suffix so that "re" doesn't match "requests"
    qual_module = f"{full_module(x)}."
    if _MAXRAY_TRANSFORM_SETTINGS.restrict_modules is not None:
        return any(
            qual_module.startswith(f"{mod}.")
            for mod in _MAXRAY_TRANSFORM_SETTINGS.restrict_modules
        )

    return True


def callable_allowed_for_transform(x, ctx: NodeContext):
    if not transform_precheck(x, ctx):
        return False

    return (
        callable(x)  # super() has getset_descriptor instead of proper __dict__
        and hasattr(x, "__dict__")
        and base_module(type(x)) not in {"ctypes"}
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
    # Forbid metaclasses as they can arbitrarily modify and replace functions
    # related SOUNDNESS BUG: function wrapper not applied via decorator - track by patching functools.wraps or tracing?
    if type(x) is not type:
        # Filter out weird stuff that would get through with an isinstance check
        # e.g. loguru: _GeneratorContextManager object is not an iterator (might be a separate bug)
        return False

    if base_module(x) in _GLOBAL_SKIP_MODULES:
        return False

    # dot suffix so that "re" doesn't match "requests"
    qual_module = f"{full_module(x)}."
    if _MAXRAY_TRANSFORM_SETTINGS.restrict_modules is not None:
        return any(
            qual_module.startswith(f"{mod}.")
            for mod in _MAXRAY_TRANSFORM_SETTINGS.restrict_modules
        )

    return True


def _maxray_walker_handler(x, ctx: NodeContext):
    # 1.  logic to recursively patch callables
    # 1a. special-case callables: __init__ and __call__
    if _MAXRAY_NOCOMPILE:
        # TODO: check cache at least?
        ...
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
                        pass_scope=_MAXRAY_TRANSFORM_SETTINGS.pass_local_scopes,
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
        else:
            # TODO: fixup control flow
            match recompile_fn_with_transform(
                x,
                _maxray_walker_handler,
                triggered_by_node=ctx,
                pass_scope=_MAXRAY_TRANSFORM_SETTINGS.pass_local_scopes,
            ):
                case Ok(x_trans):
                    # NOTE: x_trans now has _MAXRAY_TRANSFORMED field to True
                    with_fn = FunctionStore.get(x_trans._MAXRAY_TRANSFORM_ID)

                    # This does not apply when accessing X.method - only X().method
                    if (
                        inspect.ismethod(x)
                        and with_fn.data.method_info.is_inspect_method is not True
                    ):
                        logger.warning(
                            "Inconsistent method status - probable result of wrapping or metaclass shenanigans"
                        )
                        x_patched = x
                    elif inspect.ismethod(x):
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
                            set_property_on_functionlike(x, "_MAXRAY_NOTRANSFORM", True)
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


T = TypeVar("T", bound=Callable)


def maxray(
    writer: Callable[[Any, NodeContext], Any],
    *,
    mutable=True,
    forbid_modules=frozenset(),
    restrict_modules=None,
    pass_scope=False,
    initial_scope={},
    assume_transformed=False,
) -> Callable[[T], T]:
    """
    A transform that recursively hooks into all further calls made within the function, so that `writer` will (in theory) observe every single expression evaluated by the Python interpreter occurring as part of the decorated function call.

    There are some limitations to be aware of:
    - Be careful to avoid infinite recursion: the source code of the writer will not be transformed but it may call methods that have been monkey-patched that result in more calls to the writer function.
    - Objects that are not yet fully initialised may not behave as expected - e.g. repr may throw an error because of a missing property
    """

    ACTIVE_FLAG = ContextVar(f"maxray_active for <{writer}>", default=False)
    WRITER_ACTIVE_FLAG = ContextVar(f"writer_active for <{writer}>", default=False)

    global _MAXRAY_TRANSFORM_SETTINGS
    _MAXRAY_TRANSFORM_SETTINGS = _MAXRAY_TRANSFORM_SETTINGS.update(
        forbid_modules=forbid_modules,
        restrict_modules=restrict_modules,
        pass_local_scopes=pass_scope,
    )

    # TODO: allow configuring injection of variables into exec scope
    caller_locals = initial_scope

    def recursive_transform(fn):
        _MAXRAY_REGISTERED_HOOKS.append(
            W_erHook(
                writer,
                ACTIVE_FLAG,
                WRITER_ACTIVE_FLAG,
                mutable=mutable,
            )
        )

        # Fixes `test_double_decorators_with_locals`: repeated transforms are broken because stuffing closures into locals doesn't work the second time around
        if hasattr(fn, "_MAXRAY_TRANSFORM_ID") or assume_transformed:
            fn_transform = fn
        else:
            match recompile_fn_with_transform(
                fn,
                _maxray_walker_handler,
                override_scope=caller_locals,
                pass_scope=_MAXRAY_TRANSFORM_SETTINGS.pass_local_scopes,
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

        fn_with_context_update._MAXRAY_TRANSFORM_ID = fn_transform._MAXRAY_TRANSFORM_ID

        # If we're given a bound method we need to return a bound method on the same instance
        # Can only happen via xray(...)(some_method), not when applied via decorator
        if inspect.ismethod(fn):
            parent_cls = fn.__self__
            if not isinstance(parent_cls, type):
                parent_cls = type(parent_cls)

            fn_with_context_update = fn_with_context_update.__get__(
                fn.__self__, parent_cls
            )

        return fn_with_context_update

    return recursive_transform  # type: ignore
