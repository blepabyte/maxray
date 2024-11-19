import pandas as pd


def concat_dataframes():
    df_a = pd.DataFrame([{"a": 3, "b": 4}, {"a": 5, "b": 100}])
    df_b = pd.DataFrame([{"a": 3, "b": 4}, {"a": 5, "b": 100}])

    return pd.concat([df_a, df_b])


def dbg(x):
    return x


res = dbg(concat_dataframes())
print(res["b"].dtype)
assert res["b"].sum() == 208


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
assert (group_dataframes() == group_dataframes()).all().all()
