from maxray.runner.exec_sources import (
    reloadable_from_spec,
    ReloadableFunction,
    ReloadableInstance,
)


def test_code_spec():
    rl = reloadable_from_spec("lambda x, ray: x + 1").unwrap()
    assert isinstance(rl, ReloadableFunction)
    assert rl.function(1, ...) == 2


def test_module_spec():
    rl = reloadable_from_spec("maxray.inators.callgraph:Draw").unwrap()
    assert isinstance(rl, ReloadableInstance)
    print(rl.instance)


# def test_script_spec():
#     rl = reloadable_from_spec("maxray.inators.callgraph:Inator")
#     print(rl.instance)
