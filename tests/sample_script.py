if __name__ == "__main__":
    items = []
    items.append({"a": 0})
    val = items.pop()["a"]

    if val == 0:
        raise ValueError()
else:
    raise RuntimeError(f"Should be run in main context: but got {__name__}")
