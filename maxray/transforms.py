import ast
import inspect
import sys

from textwrap import dedent
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy
from result import Result, Ok, Err
from contextvars import ContextVar
from functools import wraps

from typing import Any, Callable

from loguru import logger


_METHOD_MANGLE_NAME = "vdxivosjdovs_method"


def mangle_name(identifier):
    raise NotImplementedError()


def unmangle_name(identifier):
    raise NotImplementedError()


@dataclass
class FnContext:
    impl_fn: Callable
    name: str
    module: str
    source: str
    source_file: str
    call_count: ContextVar[int]

    def __repr__(self):
        # Call count not included in repr so the same source location can be "grouped by" over multiple calls
        return f"{self.module}/{self.name}"


@dataclass
class NodeContext:
    id: str
    """
    Identifier for the type of syntax node this event came from. For example:
    - name/x
    - call/foo
    """

    source: str

    fn_context: FnContext
    """
    Properties of the function containing this node.
    """

    location: tuple[int, int, int, int]

    local_scope: Any = None

    def __repr__(self):
        return f"{self.fn_context}/{self.id}"


class FnRewriter(ast.NodeTransformer):
    def __init__(
        self,
        transform_fn,
        fn_context: FnContext,
        *,
        instance_type: str | None,
        dedent_chars: int = 0,
        record_call_counts: bool = True,
        pass_locals_on_return: bool = False,
    ):
        """
        If we're transforming a method, instance type should be the __name__ of the class. Otherwise, None.
        """

        self.transform_fn = transform_fn
        self.fn_context = fn_context
        self.instance_type = instance_type
        self.dedent_chars = dedent_chars
        self.record_call_counts = record_call_counts
        self.pass_locals_on_return = pass_locals_on_return

        # the first `def` we encounter is the one that we're transforming. Subsequent ones will be nested/within class definitions.
        self.fn_count = 0

    def is_method(self):
        return self.instance_type is not None

    @staticmethod
    def safe_unparse(node):
        # workaround for https://github.com/python/cpython/issues/108469 (fixed in python 3.12)
        try:
            return ast.unparse(node)
        except ValueError as e:
            return "<UNUNPARSEABLE>"

    @staticmethod
    def safe_show_ast(node):
        return ast.dump(node, indent=4)

    @staticmethod
    def is_private_class_name(identifier_name: str):
        return (
            identifier_name.startswith("__")
            and not identifier_name.endswith("__")
            and identifier_name.strip("_")
        )

    def recover_source(self, pre_node):
        segment = ast.get_source_segment(self.fn_context.source, pre_node, padded=False)
        if segment is None:
            return self.safe_unparse(pre_node)
        return segment

    def build_transform_node(self, node, label, node_source=None, pass_locals=False):
        """
        Builds the "inspection" node that wraps the original source node - passing the (value, context) pair to `transform_fn`.
        """
        if node_source is None:
            node_source = self.safe_unparse(node)

        line_offset = self.fn_context.impl_fn.__code__.co_firstlineno - 2
        col_offset = self.dedent_chars

        context_args = [
            ast.Constant(label),
            ast.Constant(node_source),
            # Name is injected into the exec scope by `recompile_fn_with_transform`
            ast.Name(id="_MAXRAY_FN_CONTEXT", ctx=ast.Load()),
            ast.Constant(
                (
                    line_offset + node.lineno,
                    line_offset + node.end_lineno,
                    node.col_offset + col_offset,
                    node.end_col_offset + col_offset,
                )
            ),
        ]

        if pass_locals:
            context_args.append(
                ast.Call(
                    func=ast.Name(id="_MAXRAY_BUILTINS_LOCALS", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                ),
            )
        context_node = ast.Call(
            func=ast.Name(id=NodeContext.__name__, ctx=ast.Load()),
            args=context_args,
            keywords=[],
        )

        return ast.Call(
            func=ast.Name(id=self.transform_fn.__name__, ctx=ast.Load()),
            args=[node, context_node],
            keywords=[],
        )

    def visit_Name(self, node):
        source_pre = self.recover_source(node)

        match node.ctx:
            case ast.Load():
                # Variable is accessed
                new_node = self.generic_visit(node)
            case ast.Store():
                # Variable is assigned to
                return node
            case _:
                logger.error(f"Unknown context {node.ctx}")
                return node

        return self.build_transform_node(
            new_node, f"name/{node.id}", node_source=source_pre
        )

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """
        https://docs.python.org/3/reference/expressions.html#atom-identifiers
        > Private name mangling: When an identifier that textually occurs in a class definition begins with two or more underscore characters and does not end in two or more underscores, it is considered a private name of that class. Private names are transformed to a longer form before code is generated for them. The transformation inserts the class name, with leading underscores removed and a single underscore inserted, in front of the name. For example, the identifier __spam occurring in a class named Ham will be transformed to _Ham__spam. This transformation is independent of the syntactical context in which the identifier is used. If the transformed name is extremely long (longer than 255 characters), implementation defined truncation may happen. If the class name consists only of underscores, no transformation is done.
        """
        source_pre = self.recover_source(node)

        if self.is_method() and self.is_private_class_name(node.attr):
            node.attr = f"_{self.instance_type}{node.attr}"
            logger.warning("Replaced with mangled private name")

        if isinstance(node.ctx, ast.Load):
            node = self.generic_visit(node)
            node = self.build_transform_node(
                node, f"attr/{node.attr}", node_source=source_pre
            )
        return node

    def visit_match_case(self, node: ast.match_case) -> Any:
        # leave node.pattern unchanged because the rules of match patterns are different from the rest of Python
        # throws "ValueError: MatchClass cls field can only contain Name or Attribute nodes." in compile because `case _wrap(str()):` doesn't work
        node.body = [self.generic_visit(child) for child in node.body]
        return node

    def visit_Assign(self, node: ast.Assign) -> Any:
        new_node = self.generic_visit(node)
        assert isinstance(new_node, ast.Assign)
        # node = new_node
        # node.value = self.build_transform_node(new_node, f"assign/(multiple)")
        return node

    def visit_Return(self, node: ast.Return) -> Any:
        node_pre = deepcopy(node)

        if node.value is None:
            node.value = ast.Constant(None)

        # Note: For a plain `return` statement, there's no source for a thing that *isn't* returned
        value_source_pre = self.recover_source(node.value)

        node = self.generic_visit(node)

        # TODO: Check source locations are correct here
        ast.fix_missing_locations(node)
        node.value = self.build_transform_node(
            node.value,
            f"return/{value_source_pre}",
            node_source=value_source_pre,
            pass_locals=self.pass_locals_on_return,
        )

        return ast.copy_location(node, node_pre)

    def visit_Call(self, node):
        source_pre = self.recover_source(node)

        node_pre = deepcopy(node)

        node = self.generic_visit(node)  # mutates

        # the function/callable instance itself is observed by Name/Attribute/... nodes
        return ast.copy_location(
            self.build_transform_node(
                node, f"call/{source_pre}", node_source=source_pre
            ),
            node_pre,
        )

        id_key = f"{self.context_fn.__name__}/call/{source_pre}"
        # Observes the *output* of the function
        call_observer = ast.Call(
            func=ast.Name(id=self.transform_fn.__name__, ctx=ast.Load()),
            args=[node, ast.Constant({"id": id_key})],
            keywords=[],
        )
        return call_observer

    def visit_FunctionDef(self, node: ast.FunctionDef):
        pre_node = deepcopy(node)
        self.fn_count += 1

        # Only overwrite the name of our "target function"
        if self.fn_count == 1 and self.is_method():
            node.name = f"{node.name}_{_METHOD_MANGLE_NAME}_{node.name}"

        # TODO: add is_root arg (whether fn has directly been decorated with @*xray)

        # Decorators are evaluated sequentially: decorators applied *before* our one (should?) get ignored while decorators applied *after* work correctly
        is_transform_root = self.fn_count == 1 and any(
            isinstance(decor, ast.Call)
            and isinstance(decor.func, ast.Name)
            and decor.func.id in {"maxray", "xray", "transform"}
            for decor in node.decorator_list
        )

        if is_transform_root:
            logger.info(
                f"Wiped decorators at level {self.fn_count} for {self.fn_context.impl_fn}: {node.decorator_list}"
            )
            # If we didn't clear, decorators would be applied twice - screwing up routing handling in libraries like `quart`: `@app.post("/generate")`
            node.decorator_list = []

        # Removes type annotations from the call for safety as they're evaluated at definition-time rather than call-time
        # Necessary because some packages do `if typing.TYPE_CHECKING` imports
        for arg in node.args.args:
            arg.annotation = None

        for arg in node.args.kwonlyargs:
            arg.annotation = None

        node.returns = None

        out = ast.copy_location(self.generic_visit(node), pre_node)

        # Add statements after visiting so that walk handlers aren't called on this internal code
        if self.fn_count == 1 and self.record_call_counts:
            # has to be applied as a decorator so that inner recursive calls of the fn are tracked properly
            node.decorator_list.append(
                ast.Call(
                    func=ast.Name(id="_MAXRAY_DECORATE_WITH_COUNTER", ctx=ast.Load()),
                    args=[ast.Name(id="_MAXRAY_CALL_COUNTER", ctx=ast.Load())],
                    keywords=[],
                )
            )

        return out

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        """
        Copied directly from FunctionDef
        TODO: FIX OR REFACTOR
        """

        pre_node = deepcopy(node)

        self.fn_count += 1
        # Only overwrite the name of our "target function"
        if self.fn_count == 1 and self.is_method():
            node.name = f"{node.name}_{_METHOD_MANGLE_NAME}_{node.name}"

        is_transform_root = self.fn_count == 1 and any(
            isinstance(decor, ast.Call)
            and isinstance(decor.func, ast.Name)
            and decor.func.id in {"maxray", "xray", "transform"}
            for decor in node.decorator_list
        )

        if is_transform_root:
            logger.debug(
                f"Wiped decorators at level {self.fn_count} for {self.fn_context.impl_fn}: {node.decorator_list}"
            )
            self.fn_count += 1
            node.decorator_list = []

        # Removes type annotations from the call for safety as they're evaluated at definition-time rather than call-time
        # This may not be needed now that locals are (usually) captured properly
        for arg in node.args.args:
            arg.annotation = None

        out = ast.copy_location(self.generic_visit(node), pre_node)
        return out


_TRANSFORM_CACHE = {}


def get_fn_name(fn):
    """
    Get a printable representation of the function for human-readable errors
    """
    if hasattr(fn, "__name__"):
        name = fn.__name__
    else:
        try:
            name = repr(fn)
        except Exception:
            name = "<unrepresentable function>"

    return f"{name} @ {id(fn)}"


def recompile_fn_with_transform(
    source_fn,
    transform_fn,
    ast_pre_callback=None,
    ast_post_callback=None,
    initial_scope={},
    pass_scope=False,
) -> Result[Callable, str]:
    """
    Recompiles `source_fn` so that essentially every node of its AST tree is wrapped by a call to `transform_fn` with the evaluated value along with context information about the source code.
    """
    # handle `functools.wraps`
    if hasattr(source_fn, "__wrapped__"):
        # SOUNDNESS: failure when decorators aren't applied at the definition site (will look for the original definition, ignoring any transformations that have been applied before the wrap but after definition)
        source_fn = source_fn.__wrapped__

    try:
        original_source = inspect.getsource(source_fn)

        # nested functions have excess indentation preventing compile; inspect.cleandoc(source) is an alternative but less reliable
        source = dedent(original_source)

        # want to map back to correct original location
        dedent_chars = original_source.find("\n") - source.find("\n")

        sourcefile = inspect.getsourcefile(source_fn)
        module = inspect.getmodule(source_fn)

        # the way numpy implements its array hooks means it does its own voodoo code generation resulting in functions that have source code, but no corresponding source file
        # e.g. the source file of `np.unique` is <__array_function__ internals>
        if sourcefile is None or not Path(sourcefile).exists():
            return Err(
                f"Non-existent source file ({sourcefile}) for function {get_fn_name(source_fn)}"
            )

        fn_ast = ast.parse(source)
    except OSError:
        return Err(f"No source code for function {get_fn_name(source_fn)}")
    except TypeError:
        return Err(
            f"No source code for probable built-in function {get_fn_name(source_fn)}"
        )

    # TODO: use non-overridable __getattribute__ instead?
    if not hasattr(source_fn, "__name__"):  # Safety check against weird functions
        return Err(f"There is no __name__ for function {get_fn_name(source_fn)}")

    if "super()" in source:
        # TODO: we could replace calls to super() with super(__class__, self)?
        return Err(
            f"Function {get_fn_name(source_fn)} cannot be transformed because it calls super()"
        )

    match fn_ast:
        case ast.Module(body=[ast.FunctionDef() | ast.AsyncFunctionDef()]):
            # Good
            pass
        case _:
            return Err(
                f"The targeted function {get_fn_name(source_fn)} does not correspond to a single `def` block so cannot be transformed safely!"
            )

    if ast_pre_callback is not None:
        ast_pre_callback(fn_ast)

    fn_is_method = inspect.ismethod(source_fn)
    if fn_is_method:
        # Many potential unintended side-effects
        match source_fn.__self__:
            case type():
                # Descriptor
                parent_cls = source_fn.__self__
            case _:
                # Bound method
                parent_cls = type(source_fn.__self__)

    fn_call_counter = ContextVar("maxray_call_counter", default=0)
    fn_context = FnContext(
        source_fn,
        source_fn.__name__,
        module.__name__,
        source,
        sourcefile,
        fn_call_counter,
    )
    transformed_fn_ast = FnRewriter(
        transform_fn,
        fn_context,
        instance_type=parent_cls.__name__ if fn_is_method else None,
        dedent_chars=dedent_chars,
        pass_locals_on_return=pass_scope,
    ).visit(fn_ast)
    ast.fix_missing_locations(transformed_fn_ast)

    if ast_post_callback is not None:
        ast_post_callback(transformed_fn_ast)

    scope = {
        transform_fn.__name__: transform_fn,
        NodeContext.__name__: NodeContext,
        "_MAXRAY_FN_CONTEXT": fn_context,
        "_MAXRAY_CALL_COUNTER": fn_call_counter,
        "_MAXRAY_DECORATE_WITH_COUNTER": count_calls_with,
        "_MAXRAY_BUILTINS_LOCALS": locals,
    }
    scope.update(initial_scope)

    # Add class-private names to scope (though only should be usable as a default argument)
    # TODO: should apply to all definitions within a class scope - so @staticmethod descriptors as well...
    if fn_is_method:
        scope.update(
            {
                name: val
                for name, val in parent_cls.__dict__.items()
                # TODO: BUG: ah... this excludes torch modules, right?
                if not callable(val)
            }
        )

    def extract_cell(cell):
        try:
            return cell.cell_contents
        except ValueError:
            # Cell is empty
            logger.warning(
                f"No contents for closure cell in function {get_fn_name(source_fn)} - this can happen with recursion"
            )
            return None

    scope.update(vars(module))

    if hasattr(source_fn, "__closure__") and source_fn.__closure__ is not None:
        scope.update(
            {
                name: extract_cell(cell)
                for name, cell in zip(
                    source_fn.__code__.co_freevars, source_fn.__closure__
                )
            }
        )

    if not fn_is_method and source_fn.__name__ in scope:
        logger.warning(
            f"Name {source_fn.__name__} already exists in scope for non-method"
        )

    try:
        exec(
            compile(
                transformed_fn_ast, filename=f"<{source_fn.__name__}>", mode="exec"
            ),
            scope,
            scope,
        )
    except Exception as e:
        logger.exception(e)
        logger.error(
            f"Failed to compile function `{source_fn.__name__}` at '{sourcefile}' in its module {module}"
        )
        logger.debug(f"Relevant original source code:\n{source}")
        logger.debug(f"Corresponding AST:\n{FnRewriter.safe_show_ast(fn_ast)}")
        logger.debug(
            f"Transformed code we attempted to compile:\n{FnRewriter.safe_unparse(transformed_fn_ast)}"
        )

        # FALLBACK: in numpy.core.numeric, they define `@set_module` that rewrites __module__ so inspect gives us the wrong module to correctly re-execute the def in
        # sourcefile is still correct so let's try use `sys.modules`

        file_to_modules = {
            getattr(mod, "__file__", None): mod for mod in sys.modules.values()
        }
        if sourcefile in file_to_modules:
            scope.update(vars(file_to_modules[sourcefile]))
            try:
                exec(
                    compile(
                        transformed_fn_ast,
                        filename=f"<{source_fn.__name__}>",
                        mode="exec",
                    ),
                    scope,
                    scope,
                )
            except Exception as e:
                logger.exception(e)
                return Err(
                    f"Re-def of function {get_fn_name(source_fn)} in its source file module at {sourcefile} also failed"
                )
        else:
            return Err(
                f"Failed to re-def function {get_fn_name(source_fn)} and its source file {sourcefile} was not found in sys.modules"
            )

    if fn_is_method:
        transformed_fn = scope[
            f"{source_fn.__name__}_{_METHOD_MANGLE_NAME}_{source_fn.__name__}"
        ]
    else:
        transformed_fn = scope[source_fn.__name__]

    # a decorator doesn't actually have to return a function! (could be used solely for side effect) e.g. `@register_backend_lookup_factory` for `find_content_backend` in `awkward/contents/content.py`
    if not callable(transformed_fn):
        return Err(
            f"Resulting transform of definition of {get_fn_name(source_fn)} is not even callable (got {transform_fn}). Perhaps a decorator that returns None?"
        )

    # unmangle the name again - it's possible some packages might use __name__ internally for registries and whatnot
    transformed_fn.__name__ = source_fn.__name__

    # way to keep track of which functions we've already transformed
    transformed_fn._MAXRAY_TRANSFORMED = True

    return Ok(transformed_fn)


def count_calls_with(counter: ContextVar):
    def inner(fn):
        # TODO: synchronisation/context?
        total_calls_count = 0

        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def fn_with_counter(*args, **kwargs):
                nonlocal total_calls_count
                total_calls_count += 1
                reset_call = counter.set(total_calls_count)
                try:
                    return await fn(*args, **kwargs)
                finally:
                    counter.reset(reset_call)
        else:

            @wraps(fn)
            def fn_with_counter(*args, **kwargs):
                nonlocal total_calls_count
                total_calls_count += 1
                reset_call = counter.set(total_calls_count)
                try:
                    return fn(*args, **kwargs)
                finally:
                    counter.reset(reset_call)

        return fn_with_counter

    return inner
