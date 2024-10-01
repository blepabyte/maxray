from .nodes import NodeContext, FnContext
from .function_store import (
    FunctionData,
    prepare_function_for_transform,
    get_fn_name,
    set_property_on_functionlike,
)

import ast
import inspect
import sys
import builtins

from copy import deepcopy
from result import Result, Ok, Err
from contextvars import ContextVar
from functools import wraps

from typing import Any, Callable, Optional

from loguru import logger


class RewriteRuntimeHelper:
    """
    Implementation for rewrite functionality that needs to execute code and maintain state at runtime (rather than only applying known AST transforms)

    - `read_*` are called at runtime
    - `write_*` are called during AST rewrite
    """

    SYMBOL = "_MAXRAY_REWRITE_RUNTIME"

    def __init__(self, fn_context: FnContext):
        self.fn_context = fn_context

    def expand_scope(self):
        # TODO: exclude these methods from being patched/transformed
        return {
            self.SYMBOL: self,
            "_MAXRAY_INNER_NOTRANSFORM": self.read_inner_notransform,
            "_MAXRAY_PATCH_MRO": self.read_patch_mro,
        }

    def read_locals(self):
        # This will work properly only from Python 3.13, returning a proper (mutable) view instead of a copy
        # See [PEP667](https://peps.python.org/pep-0667/)
        return sys._getframe(1).f_locals

    def write_locals(self):
        return ast.Call(
            func=ast.Attribute(
                ast.Name(id=self.SYMBOL, ctx=ast.Load()), "read_locals", ctx=ast.Load()
            ),
            args=[],
            keywords=[],
        )

    def read_inner_notransform(self, f):
        set_property_on_functionlike(f, "_MAXRAY_NOTRANSFORM", True)
        return f

    def write_inner_notransform(self):
        return ast.Name("_MAXRAY_INNER_NOTRANSFORM", ctx=ast.Load())

    def read_patch_mro(self, super_type: "super"):  # type: ignore
        # TODO: use `ctx` to find current method to patch in addition to dunders, also apply actual transform
        for parent_type in super_type.__self_class__.mro():
            if not hasattr(parent_type, "__init__") or hasattr(
                parent_type.__init__, "_MAXRAY_TRANSFORMED"
            ):  # Seems to have side effects when picked up for patching?
                continue

        return super_type

    def write_patch_mro(self):
        return ast.Name("_MAXRAY_PATCH_MRO", ctx=ast.Load())


class RewriteFailed(Exception): ...


class RewriteTransformCall(ast.Call):
    @staticmethod
    def build(
        transform_func_name: str, source_node, node_context_args, node_context_kwargs
    ):
        context_node = ast.Call(
            func=ast.Name(id=NodeContext.__name__, ctx=ast.Load()),
            args=node_context_args,
            keywords=node_context_kwargs,
        )

        return RewriteTransformCall(
            func=ast.Name(id=transform_func_name, ctx=ast.Load()),
            args=[source_node, context_node],
            keywords=[],
        )

    def assigned(self, assign_targets: list[str]):
        self.args[1] = ast.Call(
            ast.Attribute(self.args[1], "_set_assigned", ctx=ast.Load()),
            args=[ast.List([ast.Constant(s) for s in assign_targets], ctx=ast.Load())],
            keywords=[],
        )

    def iterated(self, iter_target: str):
        self.args[1] = ast.Call(
            ast.Attribute(self.args[1], "_set_iterated", ctx=ast.Load()),
            args=[ast.Constant(iter_target)],
            keywords=[],
        )

    def returned(self, return_target: str):
        self.args[1] = ast.Call(
            ast.Attribute(self.args[1], "_set_returned", ctx=ast.Load()),
            args=[ast.Constant(return_target)],
            keywords=[],
        )

    def called(self, call_args: list[str], call_kwargs: dict[str, str]):
        self.args[1] = ast.Call(
            ast.Attribute(self.args[1], "_set_called", ctx=ast.Load()),
            args=[
                ast.List([ast.Constant(a) for a in call_args], ctx=ast.Load()),
                ast.Dict(
                    keys=[ast.Constant(k) for k in call_kwargs.keys()],
                    values=[ast.Constant(v) for v in call_kwargs.values()],
                ),
            ],
            keywords=[],
        )

    def entered(self, source, as_var):
        self.args[1] = ast.Call(
            ast.Attribute(self.args[1], "_set_entered", ctx=ast.Load()),
            args=[ast.Constant(source), ast.Constant(as_var)],
            keywords=[],
        )


class FnRewriter(ast.NodeTransformer):
    def __init__(
        self,
        transform_fn,
        fn_context: FnContext,
        runtime_helper: RewriteRuntimeHelper,
        *,
        instance_type: str | None,
        dedent_chars: int = 0,
        record_call_counts: bool = True,
        pass_locals_to_ctx: bool = False,
        is_maxray_root: bool = False,
    ):
        """
        If we're transforming a method, instance type should be the __name__ of the class. Otherwise, None.
        """

        self.transform_fn = transform_fn
        self.fn_context = fn_context
        self.runtime_helper = runtime_helper
        self.instance_type = instance_type
        self.dedent_chars = dedent_chars
        self.record_call_counts = record_call_counts
        self.pass_locals_to_ctx = pass_locals_to_ctx
        self.is_maxray_root = is_maxray_root

        # the first `def` we encounter is the one that we're transforming. Subsequent ones will be nested/within class definitions.
        self.fn_count = 0
        self.known_globals_stack = []

        # function name to use to extract rewritten function from scope after executing the def
        # for methods, the name will be mangled (since they shouldn't be accessible in the module namespace)
        # for functions, will return the plain name (which might not be the __name__ of the obtained `source_fn` as __name__ can be set arbitrarily)
        self.defined_fn_name = None

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
            logger.warning(f"No source segment for {self.fn_context}")
            logger.info(self.safe_unparse(pre_node))
            return self.safe_unparse(pre_node)
        return segment

    def build_transform_node(self, node, label, node_source=None, extra_kwargs=None):
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

        if self.pass_locals_to_ctx:
            keyword_args.append(
                ast.keyword(
                    arg="local_scope",
                    value=self.runtime_helper.write_locals(),
                )
            )

        return RewriteTransformCall.build(
            self.transform_fn.__name__, node, context_args, keyword_args
        )

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

        # does the ast.Load() check need to be pulled up here?
        if self.is_private_class_name(node.attr):
            # currently we do a bad job of actually checking if it's supposed to be a method-like so this is just a hopeful guess

            if self.instance_type is not None:
                resolve_type_name = self.instance_type
            else:
                # TODO: replace with runtime getattr
                qualname_components = self.fn_context.name.split(".")
                if len(qualname_components) < 2:
                    raise RewriteFailed(
                        f"{qualname_components} :: {self.safe_unparse(node)} - couldn't guess a type to unmangle private name"
                    )
                resolve_type_name = qualname_components[-2]

            node.attr = f"_{resolve_type_name.lstrip('_')}{node.attr}"
            logger.warning(f"Replaced with mangled private name: {node.attr}")

        if isinstance(node.ctx, ast.Load):
            node = self.generic_visit(node)
            node = self.build_transform_node(
                node, f"attr/{node.attr}", node_source=source_pre
            )
        return node

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        if isinstance(node.ctx, ast.Load):
            source_pre = self.recover_source(node)
            node_pre = deepcopy(node)
            node = self.generic_visit(node)
            node = self.build_transform_node(node, "subscript", node_source=source_pre)
            return ast.copy_location(node, node_pre)
        return node

    def visit_Constant(self, node: ast.Constant) -> Any:
        source_pre = self.recover_source(node)
        node_pre = deepcopy(node)
        new_node = self.generic_visit(node)
        new_node = self.build_transform_node(
            new_node, "constant", node_source=source_pre
        )
        return ast.copy_location(new_node, node_pre)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        source_pre = self.recover_source(node)
        node_pre = deepcopy(node)
        node = self.generic_visit(node)

        op_name = type(node.op).__name__
        node = self.build_transform_node(
            node, f"binop/{op_name}", node_source=source_pre
        )
        return ast.copy_location(node, node_pre)

    def visit_match_case(self, node: ast.match_case) -> Any:
        # leave node.pattern unchanged because the rules of match patterns are different from the rest of Python
        # throws "ValueError: MatchClass cls field can only contain Name or Attribute nodes." in compile because `case _wrap(str()):` doesn't work
        node.body = [self.generic_visit(child) for child in node.body]
        return node

    def visit_withitem(self, node: ast.withitem) -> Any:
        node = deepcopy(node)
        new_node = self.generic_visit(node)
        match new_node:
            case ast.withitem(context_expr=RewriteTransformCall() as rtc):
                if node.optional_vars is None:
                    as_var = None
                else:
                    as_var = self.recover_source(node.optional_vars)
                rtc.entered(self.recover_source(node.context_expr), as_var)

        return new_node

    # Non-expression nodes

    def visit_Assign(self, node: ast.Assign) -> Any:
        node = deepcopy(node)
        new_node = self.generic_visit(node)
        match new_node:
            case ast.Assign(targets=targets, value=RewriteTransformCall() as rtc):
                target_reprs = [self.recover_source(t) for t in targets]
                rtc.assigned(target_reprs)

        return new_node

    def visit_For(self, node: ast.For) -> Any:
        node = deepcopy(node)
        new_node = self.generic_visit(node)
        match new_node:
            case ast.For(target=target, iter=RewriteTransformCall() as rtc):
                rtc.iterated(self.recover_source(target))

        return new_node

    def visit_Return(self, node: ast.Return) -> Any:
        """
        `return` is a non-expression node. Though adding an event on this node is redundant (callback would already be invoked on the expression to be returned), it's still useful to track what was returned or override it.
        """
        if node.value is None:
            value_source = ""
        else:
            value_source = self.recover_source(node.value)

        new_node = self.generic_visit(node)
        match new_node:
            case ast.Return(value=RewriteTransformCall() as rtc):
                rtc.returned(value_source)

        return new_node

    @staticmethod
    def temp_binding(node):
        """
        Useful to inspect/cache/transform some callable before calling it.

        Returns: (
            wrap_node: replacement expr for `node` that sets a temporary via a local variable
            get_node: expr retrieving the last set value of `node` without having to re-evaluate it)
        )
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
        """
        A prior approach used a walrus expression to assign to a temporary but:
        > Cannot use walrus expressions in nested list comprehension context
        """
        return wrap_node, get_node

    def visit_Call(self, node):
        source_pre = self.recover_source(node)
        node_pre = deepcopy(node)
        node = deepcopy(node)

        # Handles magic super() calls - not totally sure this is correct...
        # TODO: maybe assign the super() proxy and do the MRO patching at the start of the function
        match node:
            case ast.Call(func=ast.Name(id="globals"), args=[]):
                return ast.Name("_MAXRAY_MODULE_GLOBALS", ctx=ast.Load())

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
                    func=self.runtime_helper.write_patch_mro(),
                    args=[node],
                    keywords=[],
                )
                ast.fix_missing_locations(node)

        node_args = [self.recover_source(arg) for arg in node.args]
        node_kwargs = {kw.arg: self.recover_source(kw.value) for kw in node.keywords}

        node = self.generic_visit(node)

        match node:
            case ast.Call(func=RewriteTransformCall() as rtc):
                rtc.called(
                    node_args,
                    node_kwargs,
                )

        # the function/callable instance itself is observed by Name/Attribute/... nodes
        return ast.copy_location(
            self.build_transform_node(
                node,
                f"call/{source_pre}",
                node_source=source_pre,
            ),
            node_pre,
        )

    def transform_function_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        self.known_globals_stack.append(set())

        pre_node = deepcopy(node)
        node = deepcopy(node)
        self.fn_count += 1

        # Only overwrite the name of our "target function"
        if self.fn_count == 1:
            if self.is_method():
                node.name = f"{node.name}_{self.instance_type}_{node.name}"
            self.defined_fn_name = node.name
        else:
            node.decorator_list.insert(0, self.runtime_helper.write_inner_notransform())

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

        # avoid transforming decorators
        node.body = [self.visit(stmt) for stmt in node.body]
        out = node

        # mutable storage location for `temp_binding`
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

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return self.transform_function_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return self.transform_function_def(node)


# TODO: this could just be a collections.ChainMap
class BackingDict(dict):
    """
    We want the scope of a function to be "its module scope" + "a bunch of stuff we define on top of it" - but having scope = {**vars(), **stuff} results in changes to the module not being reflected within the scope.
    """

    def __init__(self, *primary_layers, backing_layers: list[dict]):
        super().__init__()
        for p in primary_layers:
            self.update(p)

        self.backing_layers = backing_layers

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)

        for backing in self.backing_layers:
            if key in backing:
                return backing[key]
        raise KeyError(
            f"{key} not found in backing dict ({len(self.backing_layers)} layers)"
        )


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

    # handle things like `functools.wraps`
    while hasattr(source_fn, "__wrapped__"):
        # SOUNDNESS: failure when decorators aren't applied at the definition site (will look for the original definition, ignoring any transformations that have been applied before the wrap but after definition)
        source_fn = source_fn.__wrapped__

    match prepare_function_for_transform(source_fn, from_ctx=triggered_by_node):
        case Ok(with_source_fn):
            pass
        case Err(err):
            logger.error(err)
            return Err(err)

    module = lookup_module(with_source_fn)
    if module is None:
        return with_source_fn.mark_errored(
            f"Non-existent source module `{getattr(source_fn, '__module__', None)}` for function {get_fn_name(source_fn)}"
        )

    # A bit hacky: Handle non-"methods" like NDFrame.isna (unbound, not called from instance)
    if (
        special_use_instance_type is None
        and not with_source_fn.method_info.is_inspect_method
    ):
        qual_components = with_source_fn.qualname.split(".")
        match qual_components:
            case [*_, qual_cls, _fn_name] if hasattr(module, qual_cls):
                # Pretend it's a method for most purposes
                # TODO: Update Method data in store for consistency?
                special_use_instance_type = getattr(module, qual_cls)

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
    if fn_is_method:
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

    module_name = module.__name__
    use_relative_import_root = module.__name__.split(".")[0]
    use_relative_import_root = getattr(module, "__package__", use_relative_import_root)

    fn_call_counter = ContextVar("maxray_call_counter", default=0)
    fn_context = FnContext(
        source_fn,
        source_fn.__qualname__,
        module_name,
        with_source_fn.source,
        with_source_fn.source_file,
        with_source_fn.source_offset_lines,
        fn_call_counter,
        compile_id=with_source_fn.compile_id,
    )
    runtime_helper = RewriteRuntimeHelper(fn_context)

    instance_type = parent_cls.__name__ if fn_is_method else None
    fn_rewriter = FnRewriter(
        transform_fn,
        fn_context,
        runtime_helper,
        instance_type=instance_type,
        dedent_chars=with_source_fn.source_dedent_chars,
        pass_locals_to_ctx=pass_scope,
        is_maxray_root=is_maxray_root,
    )

    try:
        transformed_fn_ast = fn_rewriter.visit(fn_ast)
    except RewriteFailed as rewrite_err:
        return with_source_fn.mark_errored(
            f"{rewrite_err}: function rewrite failed in transform"
        )

    ast.fix_missing_locations(transformed_fn_ast)

    if ast_post_callback is not None:
        ast_post_callback(transformed_fn_ast)

    scope_layers = {
        "core": {
            transform_fn.__name__: transform_fn,
            NodeContext.__name__: NodeContext,
            "_MAXRAY_FN_CONTEXT": fn_context,
            "_MAXRAY_CALL_COUNTER": fn_call_counter,
            "_MAXRAY_DECORATE_WITH_COUNTER": count_calls_with,
            "_MAXRAY_MODULE_GLOBALS": vars(module),
            **runtime_helper.expand_scope(),
        },
        "override": override_scope,
        "class_local": {},
        "module": {},
        "closure": {},
    }

    # Fix relative imports within functions
    scope_layers["override"]["__package__"] = use_relative_import_root

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
            **scope_layers["core"],
            **scope_layers["closure"],
            **scope_layers["override"],
        }
        scope = BackingDict(
            scope,
            backing_layers=[vars(module), vars(builtins), scope_layers["class_local"]],
        )

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
        return with_source_fn.mark_errored(
            f"{e}: Failed to re-def function {get_fn_name(source_fn)}"
        )

    transformed_fn = scope[fn_rewriter.defined_fn_name]

    # a decorator doesn't actually have to return a function! (could be used solely for side effect) e.g. `@register_backend_lookup_factory` for `find_content_backend` in `awkward/contents/content.py`
    if not callable(transformed_fn) and not inspect.ismethoddescriptor(transformed_fn):
        return with_source_fn.mark_errored(
            f"Resulting transform of definition of {get_fn_name(source_fn)} is not even callable (got {transform_fn}). Perhaps a decorator that returns None?"
        )

    # unmangle the name again - it's possible some packages might use __name__ internally for registries and whatnot
    transformed_fn.__name__ = source_fn.__name__
    transformed_fn.__qualname__ = source_fn.__qualname__

    # way to keep track of which functions we've already transformed
    transformed_fn._MAXRAY_TRANSFORM_ID = with_source_fn.compile_id
    with_source_fn.mark_compiled(transformed_fn)

    assert set_property_on_functionlike(
        transformed_fn, "_MAXRAY_TRANSFORM_ID", with_source_fn.compile_id
    )

    transformed_fn.__module__ = module.__name__

    # TODO: unify places where transform ID is set
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
    # Prefer to match on source file rather than __module__ (which can be overriden arbitrarily, e.g. in numpy.core.numeric, they define `@set_module`)
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
