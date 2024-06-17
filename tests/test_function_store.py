from maxray.function_store import FunctionStore


def test_collect():
    df = FunctionStore.collect()
    print(df.to_pandas())
