from maxray.function_store import FunctionStore, set_property_on_functionlike


def make_class():
    class HasFun_ctions:
        def f1(self):
            pass

        @staticmethod
        def f2():
            pass

        @classmethod
        def f3(cls):
            pass

    return HasFun_ctions


def test_property_setter():
    funcs = ["f1", "f2", "f3"]

    f_class = make_class()
    f_instance = make_class()()

    counter = 0
    for f in [getattr(f_class, fn) for fn in funcs]:
        set_property_on_functionlike(f, "test", counter)
        assert getattr(f, "test") == counter
        counter += 1

    for f in [getattr(f_instance, fn) for fn in funcs]:
        set_property_on_functionlike(f, "test", counter)
        assert getattr(f, "test") == counter
        counter += 1
