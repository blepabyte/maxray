from maxray import xray, maxray
from maxray.transforms import NodeContext
from maxray.walkers import dbg

import importlib.util
import tempfile

import pytest


def if_package(pkg):
    def decorator(func):
        def wrapper(*args, **kwargs):
            spec = importlib.util.find_spec(pkg)
            if spec is None:
                pytest.skip(f"{pkg} package not installed")
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


@if_package("torch")
def test_pytorch_ops():
    import torch

    @maxray(dbg)
    def create_tensor():
        x = torch.ones(3, 3)
        y = x.to(torch.float64)
        return y.sum()

    assert create_tensor() == 9

    def eq_check(a, b):
        return a == b

    def mutate(x, ctx):
        if isinstance(x, str) and x == "a":
            return "b"
        return x

    @maxray(mutate)
    def check_cmp():
        return eq_check("a", "b")

    assert check_cmp() is True


@pytest.mark.xfail
@if_package("torch")
def test_pytorch_nograd_context():
    import torch

    # Currently throws: NameError: name 'torch' is not defined
    @torch.no_grad()
    def eq_check(a, b):
        return a == b

    def mutate(x, ctx):
        if isinstance(x, str) and x == "a":
            return "b"
        return x

    @maxray(mutate)
    def check_cmp():
        return eq_check("a", "b")

    assert check_cmp() is True


def test_pandas_dataframe_ops():
    import pandas as pd

    @xray(dbg)
    def concat_dataframes():
        df_a = pd.DataFrame([{"a": 3, "b": 4}, {"a": 5, "b": 100}])
        df_b = pd.DataFrame([{"a": 3, "b": 4}, {"a": 5, "b": 100}])

        return pd.concat([df_a, df_b])

    assert concat_dataframes()["b"].sum() == 208

    def group_dataframes():
        df = pd.DataFrame(
            [
                {"module": "foo.1", "dtype": "float32", "size": 10},
                {"module": "bar.2", "dtype": "float32", "size": 100},
            ]
        )
        df["layer"] = df.module.apply(lambda x: ".".join(x.split(".")[:-1]))

        dfg = df.groupby(
            [
                "layer",
                "dtype",
            ]
        )
        counts: pd.Series = dfg["size"].count()
        sizes: pd.Series = dfg["size"].sum()

        param_table = pd.concat(
            [
                counts.rename("counts"),
                sizes.rename("total_parameters"),
                (sizes / 1e6).rename("total_parameters/1M"),
            ],
            axis=1,
        )

        return param_table

    # Checks that the transformation doesn't change behaviour
    assert (xray(dbg)(group_dataframes)() == group_dataframes()).all().all()


def test_numpy_formats():
    import numpy as np

    @xray(dbg)
    def npy_save_load():
        X = np.eye(10)

        with tempfile.NamedTemporaryFile(suffix=".npy") as f:
            np.save(f.name, X)

            Y = np.load(f.name)

        return X, Y

    X, Y = npy_save_load()
    assert (X == Y).all()
