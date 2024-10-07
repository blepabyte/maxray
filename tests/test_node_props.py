from maxray import xray, maxray
from maxray.nodes import NodeContext, RayContext


def test_assignment_unpacking():
    collect_xy = []

    def collect_split(x, ray: RayContext):
        match ray.assigned():
            case {"x": _x, "y": _y}:
                collect_xy.extend((_x + 1, _y + 1))
            case _:
                raise ValueError()

    @xray(collect_split, preserve_values=False)
    def assign_split(xy):
        x, y = xy

    assign_split([100, 1000])

    assert collect_xy == [101, 1001]
