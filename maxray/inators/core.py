import inspect


class Statefool:
    def __init__(self):
        self._existing_keys = {}

    def __getitem__(self, key):
        return self._existing_keys[key][1]

    def __setitem__(self, key, value):
        v, _old_value = self._existing_keys[key]
        self._existing_keys[key] = (v, value)

    def define_once(self, key, factory, /, v: int = 0):
        """
        Args:
            key (Immutable + Hash + Eq): Identifies this object between steps
        """
        # Note: can use runtime value for def key for auto grouping!
        ex_o = None
        # Not thread-safe....
        if key in self._existing_keys:
            ex_v, ex_o = self._existing_keys[key]
            if ex_v >= v:
                return ex_o

        self._existing_keys[key] = (v, new_o := factory(ex_o))
        return new_o


class Matcher:
    """
    Pattern-matching helper to provide a stable interface over source information stored in NodeContext.props (collected during AST parse)
    """

    def __init__(self, x, ctx):
        self.x = x
        self.unpacked_x = x
        self.ctx = ctx

        match ctx.props:
            case {"assigned": {"targets": targets}}:
                if len(targets) > 1:
                    if inspect.isgenerator(x) or isinstance(x, (map, filter)):
                        # Greedily consume iterators before assignment
                        self.unpacked_x = tuple(iter(x))
                    else:
                        # Otherwise for chained equality like a = b, c = it, code relies on `a` being of the original type
                        self.unpacked_x = x
                    # TODO: doesn't work for starred assignments: x, *y, z = iterable
                    self._assigned = {
                        target: val for target, val in zip(targets, self.unpacked_x)
                    }
                elif len(targets) == 1:
                    self._assigned = {targets[0]: x}
                else:
                    self._assigned = {}
            case _:
                self._assigned = {}

    def __iter__(self):
        yield self.x
        yield self.ctx

    def assigned(self):
        return self._assigned

    def unpacked(self):
        return self.unpacked_x


S = Statefool()
