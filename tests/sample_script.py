if __name__ == "__main__":
    items = []
    items.append({"a": 0})
    val = items.pop()["a"]

    if val == 0:
        raise ValueError()
