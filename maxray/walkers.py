from loguru import logger


def dbg(x, ctx):
    try:
        x_repr = repr(x)
    except Exception:
        x_repr = "<UNREPRESENTABLE>"

    logger.debug(f"{ctx} :: {x_repr}")
    return x
