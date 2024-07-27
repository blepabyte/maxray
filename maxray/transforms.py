from .function_store import FunctionData, prepare_function_for_transform, get_fn_name

import ast
import inspect
import sys
import uuid
import re
import builtins

from textwrap import dedent
from pathlib import Path
from dataclasses import dataclass, asdict
from copy import deepcopy
from result import Result, Ok, Err
from contextvars import ContextVar
from functools import wraps

from typing import Any, Callable, Optional

from loguru import logger


_METHOD_MANGLE_NAME = "vdxivosjdovs_method"


def mangle_name(identifier):
    raise NotImplementedError()


def unmangle_name(identifier):
    raise NotImplementedError()


@dataclass
class FnContext:
    impl_fn: Callable
    name: str  # qualname
    module: str
    source: str
    source_file: str
    line_offset: int
    "Line number in `source_file` at which the function definition begins."

    call_count: ContextVar[int]
    compile_id: str

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
    """
    (start_line, end_line, start_col, end_col) as 0-indexed location of `source` in `fn_context.source_file`
    """

    local_scope: Any = None

    caller_id: Any = None

    def __repr__(self):
        return f"{self.fn_context}/{self.id}"

    def to_dict(self):
        return {
            "id": self.id,
            "source": self.source,
            "location": self.location,
            "caller_id": self.caller_id,
            "fn_context": {
                "name": self.fn_context.name,
                "module": self.fn_context.module,
                "source_file": self.fn_context.source_file,
                "call_count": self.fn_context.call_count.get(),
            },
        }


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
        is_maxray_root: bool = False,
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
        self.is_maxray_root = is_maxray_root

        # the first `def` we encounter is the one that we're transforming. Subsequent ones will be nested/within class definitions.
        self.fn_count = 0
        self.known_globals_stack = []

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

    def build_transform_node(
        self, node, label, node_source=None, pass_locals=False, extra_kwargs=None
    ):
        """
        Builds the "inspection" node that wraps the original source node - passing the (value, context) pair to `transform_fn`.
        """
        node = deepcopy(node)
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

        keyword_args = []

        if extra_kwargs is not None:
            keyword_args.extend(extra_kwargs)

        if pass_locals:
            keyword_args.append(
                ast.keyword(
                    arg="local_scope",
                    value=ast.Call(
                        func=ast.Name(id="_MAXRAY_BUILTINS_LOCALS", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    ),
                )
            )

        context_node = ast.Call(
            func=ast.Name(id=NodeContext.__name__, ctx=ast.Load()),
            args=context_args,
            keywords=keyword_args,
        )
        ret = ast.Call(
            func=ast.Name(id=self.transform_fn.__name__, ctx=ast.Load()),
            args=[node, context_node],
            keywords=[],
        )
        return ret

    def visit_Name(self, node):
        source_pre = self.recover_source(node)
        node = deepcopy(node)

        match node.ctx:
            case ast.Load():
                # Variable is accessed
                if node.id in self.known_globals_stack[-1]:
                    new_node = ast.fix_missing_locations(
                        ast.Subscript(
                            value=ast.Name(id="_MAXRAY_MODULE_GLOBALS", ctx=ast.Load()),
                            slice=ast.Constant(node.id),
                            ctx=ast.Load(),
                        )
                    )
                else:
                    new_node = node
            case ast.Store():
                # Variable is assigned to
                if node.id in self.known_globals_stack[-1]:
                    return ast.Subscript(
                        value=ast.Name(id="_MAXRAY_MODULE_GLOBALS", ctx=ast.Load()),
                        slice=ast.Constant(node.id),
                        ctx=ast.Store(),
                    )
                else:
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
        node = deepcopy(node)

        # if self.is_method() and self.is_private_class_name(node.attr):
        if self.is_private_class_name(node.attr):
            # currently we do a bad job of actually checking if it's supposed to be a method-like so this is just a hopeful guess
            qualname_type_guess = self.fn_context.name.split(".")[-2]
            if self.instance_type is not None:
                node.attr = f"_{self.instance_type}{node.attr}"
            else:
                node.attr = f"_{qualname_type_guess}{node.attr}"
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
        node = deepcopy(node)
        new_node = self.generic_visit(node)
        assert isinstance(new_node, ast.Assign)
        # node = new_node
        # node.value = self.build_transform_node(new_node, f"assign/(multiple)")
        return node

    def visit_Return(self, node: ast.Return) -> Any:
        node_pre = deepcopy(node)
        node = deepcopy(node)

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

    @staticmethod
    def temp_binding(node):
        """
        Returns a wrapping node that sets the temporary, and a node to retrieve the last set value.
        """
        wrap_node = ast.fix_missing_locations(
            ast.Call(
                ast.Name("_MAXRAY_SET_TEMP", ctx=ast.Load()),
                [ast.Name("_MAXRAY_TEMP_LOCAL_VAR", ctx=ast.Load()), node],
                keywords=[],
            )
        )
        get_node = ast.fix_missing_locations(
            ast.Subscript(
                ast.Name("_MAXRAY_TEMP_LOCAL_VAR", ctx=ast.Load()),
                slice=ast.Constant(0),
                ctx=ast.Load(),
            )
        )
        return wrap_node, get_node
        # Cannot use walrus expressions in nested list comprehension context
        # return ast.fix_missing_locations(
        #     ast.Subscript(
        #         value=ast.Tuple(
        #             elts=[
        #                 ast.NamedExpr(
        #                     target=ast.Name(id=by_name, ctx=ast.Store()),
        #                     value=node,
        #                 ),
        #                 ast.Name(id=by_name, ctx=ast.Load()),
        #             ],
        #             ctx=ast.Load(),
        #         ),
        #         slice=ast.Constant(-1),
        #         ctx=ast.Load(),
        #     )
        # )

    def visit_Call(self, node):
        source_pre = self.recover_source(node)
        node_pre = deepcopy(node)
        node = deepcopy(node)

        match node:
            case ast.Call(func=ast.Name(id="super"), args=[]):
                # HACK: STUPID HACKY HACK
                # TODO: detect @classmethod properly
                if "self" in self.fn_context.source:
                    node.args = [
                        ast.Name("__class__", ctx=ast.Load()),
                        ast.Name("self", ctx=ast.Load()),
                    ]
                else:
                    node.args = [
                        ast.Name("__class__", ctx=ast.Load()),
                        ast.Name("cls", ctx=ast.Load()),
                    ]
                node = ast.Call(
                    func=ast.Name("_MAXRAY_PATCH_MRO", ctx=ast.Load()),
                    args=[node],
                    keywords=[],
                )
                ast.fix_missing_locations(node)

        node = self.generic_visit(node)  # mutates
        # Want to keep track of which function we're calling
        # very very not thread-safe...
        # `node.func` is now likely a call to `_maxray_walker_handler`
        node.func, node_func_ref = self.temp_binding(node.func)
        # We can't use `node_pre.func` beacuse evaluating it has side effects

        # TODO: maybe assign the super() proxy and do the MRO patching at the start of the function
        extra_kwargs = []
        if "super" not in source_pre:
            extra_kwargs.append(
                ast.keyword(
                    "caller_id",
                    # value=ast.Name("_MAXRAY_TEMP_VAR", ctx=ast.Load()),
                    value=ast.Call(
                        func=ast.Name("getattr", ctx=ast.Load()),
                        args=[
                            node_func_ref,
                            ast.Constant("_MAXRAY_TRANSFORM_ID"),
                            ast.Constant(None),
                        ],
                        keywords=[],
                    ),
                )
            )

        # the function/callable instance itself is observed by Name/Attribute/... nodes
        return ast.copy_location(
            self.build_transform_node(
                node,
                f"call/{source_pre}",
                node_source=source_pre,
                extra_kwargs=extra_kwargs,
            ),
            node_pre,
        )

    def transform_function_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        self.known_globals_stack.append(set())

        pre_node = deepcopy(node)
        node = deepcopy(node)
        self.fn_count += 1

        # Only overwrite the name of our "target function"
        if self.fn_count == 1 and self.is_method():
            node.name = f"{node.name}_{self.instance_type}_{node.name}"

        # Decorators are evaluated sequentially: decorators applied *before* our one (should?) get ignored while decorators applied *after* work correctly
        is_transform_root = self.fn_count == 1 and self.is_maxray_root

        if is_transform_root:
            # If we didn't clear, decorators would be applied twice - screwing up routing handling in libraries like `quart`: `@app.post("/generate")`
            node.decorator_list = []
        else:
            # Handle nested @xray calls (e.g. running via `xpy` - everything inside has already been transformed but there won't be source code)
            for dec in node.decorator_list:
                match dec:
                    case ast.Call(func=ast.Name(id="maxray" | "xray")):
                        dec.keywords.append(
                            ast.keyword(
                                "assume_transformed",
                                value=ast.Constant(True),
                            )
                        )

        # Removes type annotations from the call for safety as they're evaluated at definition-time rather than call-time
        # Necessary because some packages do `if typing.TYPE_CHECKING` imports
        for arg in node.args.args:
            arg.annotation = None

        for arg in node.args.kwonlyargs:
            arg.annotation = None

        node.returns = None

        node.body = [self.visit(stmt) for stmt in node.body]
        out = node
        # out = self.generic_visit(node)

        assign_stmt = ast.Assign(
            targets=[ast.Name(id="_MAXRAY_TEMP_LOCAL_VAR", ctx=ast.Store())],
            value=ast.List(elts=[ast.Constant(value=None)], ctx=ast.Load()),
        )
        out.body.insert(0, assign_stmt)

        out = ast.copy_location(out, pre_node)

        # Add statements after visiting so that walk handlers aren't called on this internal code
        if self.fn_count == 1 and self.record_call_counts:
            # has to be applied as a decorator so that inner recursive calls of the fn are tracked properly
            out.decorator_list.append(
                ast.Call(
                    func=ast.Name(id="_MAXRAY_DECORATE_WITH_COUNTER", ctx=ast.Load()),
                    args=[ast.Name(id="_MAXRAY_CALL_COUNTER", ctx=ast.Load())],
                    keywords=[],
                )
            )

        self.known_globals_stack.pop()
        return out

    def visit_Global(self, node: ast.Global):
        # global declaration only applies at immediate function def level
        self.known_globals_stack[-1].update(node.names)
        return ast.fix_missing_locations(ast.Expr(ast.Constant(None)))
        # return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return self.transform_function_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return self.transform_function_def(node)


_TRANSFORM_CACHE = {}


def recompile_fn_with_transform(
    source_fn,
    transform_fn,
    ast_pre_callback=None,
    ast_post_callback=None,
    override_scope={},
    pass_scope=False,
    special_use_instance_type=None,
    is_maxray_root=False,
    triggered_by_node: Optional[NodeContext] = None,
) -> Result[Callable, str]:
    """
    Recompiles `source_fn` so that essentially every node of its AST tree is wrapped by a call to `transform_fn` with the evaluated value along with context information about the source code.
    """
    original_source_fn = source_fn
    match prepare_function_for_transform(source_fn, from_ctx=triggered_by_node):
        case Ok(with_source_fn):
            pass
        case Err(err):
            logger.error(err)
            return Err(err)
    # match WithFunction.try_for_transform(source_fn):
    #     case Ok((source_fn, with_source_fn)):
    #         pass
    #     case Err((err, _wf)):
    #         return Err(err)

    source_fn = original_source_fn
    # handle `functools.wraps`
    # TODO: multiple layers of nesting?
    if hasattr(source_fn, "__wrapped__"):
        # SOUNDNESS: failure when decorators aren't applied at the definition site (will look for the original definition, ignoring any transformations that have been applied before the wrap but after definition)
        source_fn = source_fn.__wrapped__

    module = lookup_module(with_source_fn)
    # module = inspect.getmodule(source_fn)
    if module is None:
        return with_source_fn.mark_errored(
            f"Non-existent source module `{getattr(source_fn, '__module__', None)}` for function {get_fn_name(source_fn)}"
        )

    try:
        fn_ast = ast.parse(with_source_fn.source)
    except SyntaxError:
        return with_source_fn.mark_errored(
            f"Syntax error in function {get_fn_name(source_fn)}"
        )

    # TODO: warn on "nonlocal" statements?

    match fn_ast:
        case ast.Module(body=[ast.FunctionDef() | ast.AsyncFunctionDef()]):
            # Good
            pass
        case _:
            return with_source_fn.mark_errored(
                f"The targeted function {get_fn_name(source_fn)} does not correspond to a single `def` block so cannot be transformed safely!"
            )

    if ast_pre_callback is not None:
        ast_pre_callback(fn_ast)

    fn_is_method = with_source_fn.method_info.is_inspect_method
    # fn_is_method = with_source_fn.method.is_inspect_method is not None
    if fn_is_method:
        # parent_cls = with_source_fn.method.instance_cls
        # parent_cls = with_source_fn.method.instance_cls
        # Many potential unintended side-effects
        match original_source_fn.__self__:
            case type():
                # Descriptor
                parent_cls = original_source_fn.__self__
            case _:
                # Bound method
                parent_cls = type(original_source_fn.__self__)

        if with_source_fn.method_info.defined_on_cls is not None:
            parent_cls = with_source_fn.method_info.defined_on_cls

    # yeah yeah an unbound __init__ isn't actually a method but we can basically treat it as one
    if special_use_instance_type is not None:
        fn_is_method = True
        parent_cls = special_use_instance_type

    fn_call_counter = ContextVar("maxray_call_counter", default=0)
    fn_context = FnContext(
        source_fn,
        source_fn.__qualname__,
        module.__name__,
        with_source_fn.source,
        with_source_fn.source_file,
        with_source_fn.source_offset_lines,
        fn_call_counter,
        compile_id=with_source_fn.compile_id,
    )
    instance_type = parent_cls.__name__ if fn_is_method else None
    transformed_fn_ast = FnRewriter(
        transform_fn,
        fn_context,
        instance_type=instance_type,
        dedent_chars=with_source_fn.source_dedent_chars,
        # dedent_chars=with_source_fn.added_context["dedent_chars"],
        pass_locals_on_return=pass_scope,
        is_maxray_root=is_maxray_root,
    ).visit(fn_ast)
    ast.fix_missing_locations(transformed_fn_ast)

    if ast_post_callback is not None:
        ast_post_callback(transformed_fn_ast)

    def patch_mro(super_type: super):
        for parent_type in super_type.__self_class__.mro():
            # Ok that's weird - this function gets picked up by the maxray decorator and seems to correctly patch the parent types - so despite looking like this function does absolutely nothing, it actually *has* side-effects
            if not hasattr(parent_type, "__init__") or hasattr(
                parent_type.__init__, "_MAXRAY_TRANSFORMED"
            ):
                continue

        return super_type

    scope_layers = {
        "core": {
            transform_fn.__name__: transform_fn,
            NodeContext.__name__: NodeContext,
            "_MAXRAY_FN_CONTEXT": fn_context,
            "_MAXRAY_CALL_COUNTER": fn_call_counter,
            "_MAXRAY_DECORATE_WITH_COUNTER": count_calls_with,
            "_MAXRAY_BUILTINS_LOCALS": locals,
            "_MAXRAY_PATCH_MRO": patch_mro,
            "_MAXRAY_MODULE_GLOBALS": vars(
                module
            ),  # TODO: on fail need to use different module
        },
        "override": override_scope,
        "class_local": {},
        "module": {},
        "closure": {},
    }

    # BUG: this will NOT work with threading - could use ContextVar if no performance impact?
    def set_temp(state, value):
        state[0] = value
        return value

    scope_layers["core"]["_MAXRAY_SET_TEMP"] = set_temp
    # Add class-private names to scope (though only should be usable as a default argument)
    # TODO: should apply to all definitions within a class scope - so @staticmethod descriptors as well...
    if fn_is_method:
        scope_layers["class_local"].update(
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

    scope_layers["module"].update(vars(module))

    if hasattr(source_fn, "__closure__") and source_fn.__closure__ is not None:
        scope_layers["closure"].update(
            {
                name: extract_cell(cell)
                for name, cell in zip(
                    source_fn.__code__.co_freevars, source_fn.__closure__
                )
            }
        )

    try:
        # TODO: this might be slow
        scope = {
            **scope_layers["class_local"],
            **vars(builtins),
            **scope_layers["core"],
            **scope_layers["module"],
            **scope_layers["closure"],
            **scope_layers["override"],
        }

        if not fn_is_method and source_fn.__name__ in scope:
            logger.warning(
                f"Name {source_fn.__name__} already exists in scope for non-method"
            )

        exec(
            compile(
                transformed_fn_ast,
                filename=f"<{source_fn.__name__} in {with_source_fn.source_file}>",
                mode="exec",
            ),
            scope,
            scope,
        )
    except Exception as e:
        logger.exception(e)
        logger.error(
            f"Failed to compile function `{source_fn.__name__}` at '{with_source_fn.source_file}' in its module {module}"
        )
        logger.trace(f"Relevant original source code:\n{with_source_fn.source}")
        logger.trace(f"Corresponding AST:\n{FnRewriter.safe_show_ast(fn_ast)}")
        logger.debug(
            f"Transformed code we attempted to compile:\n{FnRewriter.safe_unparse(transformed_fn_ast)}"
        )

        # FALLBACK: in numpy.core.numeric, they define `@set_module` that rewrites __module__ so inspect gives us the wrong module to correctly re-execute the def in
        # sourcefile is still correct so let's try use `sys.modules`

        file_to_modules = {
            getattr(mod, "__file__", None): mod for mod in sys.modules.values()
        }
        if with_source_fn.source_file in file_to_modules:
            # Re-executing in a different module: re-declare scope without the previous module (otherwise we get incorrect behaviour like `min` being replaced with `np.amin` in `np.load`)
            scope = {
                **scope_layers["class_local"],
                **vars(builtins),
                **scope_layers["core"],
                **vars(file_to_modules[with_source_fn.source_file]),
                **scope_layers["closure"],
                **scope_layers["override"],
            }

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
                return with_source_fn.mark_errored(
                    f"Re-def of function {get_fn_name(source_fn)} in its source file module at {with_source_fn.source_file} also failed"
                )
        else:
            return with_source_fn.mark_errored(
                f"Failed to re-def function {get_fn_name(source_fn)} and its source file {with_source_fn.source_file} was not found in sys.modules"
            )

    if fn_is_method:
        transformed_fn = scope[
            f"{source_fn.__name__}_{instance_type}_{source_fn.__name__}"
        ]
    else:
        transformed_fn = scope[source_fn.__name__]

    # a decorator doesn't actually have to return a function! (could be used solely for side effect) e.g. `@register_backend_lookup_factory` for `find_content_backend` in `awkward/contents/content.py`
    if not callable(transformed_fn) and not inspect.ismethoddescriptor(transformed_fn):
        return with_source_fn.mark_errored(
            f"Resulting transform of definition of {get_fn_name(source_fn)} is not even callable (got {transform_fn}). Perhaps a decorator that returns None?"
        )

    # unmangle the name again - it's possible some packages might use __name__ internally for registries and whatnot
    transformed_fn.__name__ = source_fn.__name__

    # way to keep track of which functions we've already transformed
    transformed_fn._MAXRAY_TRANSFORMED = True
    transformed_fn._MAXRAY_TRANSFORM_ID = with_source_fn.compile_id
    with_source_fn.mark_compiled(transformed_fn)

    # if hasattr(transformed_fn, "__wrapped__"):
    #     transformed_fn.__wrapped__._MAXRAY_TRANSFORMED = True
    #     transformed_fn.__wrapped__._MAXRAY_TRANSFORM_ID = (
    #         with_source_fn.compiled().compile_id
    #     )

    return Ok(transformed_fn)


# TODO: probably better to modify the generated code directly instead of relying on a wrapper...
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


FILE_TO_SYS_MODULES = {}


def lookup_module(fd: FunctionData):
    if fd.source_file in FILE_TO_SYS_MODULES:
        return FILE_TO_SYS_MODULES[fd.source_file]

    file_matched = None
    module_matched = None
    for module_name, module in sys.modules.items():
        module_path = getattr(module, "__file__", None)

        if module_path is not None:
            FILE_TO_SYS_MODULES[module_path] = module

            if module_path == fd.source_file:
                file_matched = module

        if module_name == fd.module:
            module_matched = module

    if file_matched:
        return file_matched
    else:
        return module_matched
