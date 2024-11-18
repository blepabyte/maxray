from maxray.inators.core import S, Ray
from maxray.inators.base import BaseInator
from maxray.function_store import FunctionStore
from maxray.runner import (
    MAIN_FN_NAME,
    ExecInfo,
    RunAborted,
    RunCompleted,
    RunErrored,
    AbortRun,
    RestartRun,
    Break,
)

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_agraph import write_dot
import numpy as np
import pandas as pd

from io import BytesIO, StringIO
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Literal

import rerun as rr
import rerun.blueprint as rrb

import click


class CaptureCallgraph(BaseInator):
    def __init__(self, all_calls=True):
        super().__init__()
        self.G = nx.DiGraph()
        self.all_calls = all_calls

    def graph(self) -> nx.DiGraph:
        return self.G

    def xray(self, x, ray: Ray):
        src_id = ray.ctx.fn_context.name

        # raise Break()
        # TODO: handle noncompiled and errored functions

        # TODO: identify functions properly/qualname/modules
        match ray.called():
            case {"target": _} if isinstance(x, type):
                # constructor call: __init__
                dst_id = f"{x.__name__}.__init__"
                dst_belongs_to_module = x.__module__

            case {"fn_compile_id": compile_id}:
                dst_fn = FunctionStore.get(compile_id).data
                dst_id = dst_fn.qualname
                dst_belongs_to_module = dst_fn.module

            case {"target": _} if self.all_calls:
                try:
                    dst_id = x.__getattribute__("__name__")
                    dst_belongs_to_module = x.__getattribute__("__module__")

                except Exception:
                    return
            case _:
                return

        if dst_belongs_to_module in {"builtins"}:
            return

        self.G.add_edge(src_id, dst_id)
        self.G.nodes[src_id]["module"] = self.G.nodes[src_id].get("module", "_")

        self.G.nodes[dst_id]["module"] = str(dst_belongs_to_module).split(".")[0]
        self.G.nodes[dst_id]["fontname"] = "IosevkaTerm NF"


def highlight(
    x, cmap="colorcet:CET_C8", ngroups=16, fmt: Literal["hex", "rgba"] = "rgba"
):
    from cmap import Colormap
    from hashlib import sha256

    # stdlib hash is not pure
    x_pos = (int.from_bytes(sha256(repr(x).encode()).digest()) % ngroups) / (
        ngroups - 1
    )
    color = Colormap(cmap)(x_pos)
    if fmt == "hex":
        return color.hex
    else:
        return color.rgba


class Draw(CaptureCallgraph):
    def __init__(
        self,
        backend: Literal["networkx", "rerun"],
        draw_args: dict,
        rotate: bool,
        save_figure: Optional[Path],
    ):
        super().__init__()
        self.draw_backend = backend
        if self.draw_backend == "rerun":
            rr.init("callgraph", spawn=True)

        self.rotate = rotate
        self.draw_args = draw_args
        self.save_figure = save_figure

    @staticmethod
    @click.command()
    @click.option("--labels", is_flag=True)
    @click.option("--save", type=Path)
    @click.option("--nx", "backend", flag_value="networkx", default=True)
    @click.option("--rr", "--rerun", "backend", flag_value="rerun")
    @click.option("--rotate", is_flag=True)
    def cli(
        labels: bool,
        rotate: bool,
        backend: Literal["networkx", "rerun"],
        save: Optional[Path],
    ):
        # TODO: prefer rerun if installed

        draw_args = {}
        if labels:
            draw_args["with_labels"] = True

        # TODO: post-processing view for just the __init__ dependency graph?
        return Draw(
            draw_args=draw_args, rotate=rotate, backend=backend, save_figure=save
        )

    def draw_embedded_mpl(self, pos_embeds: dict[str, tuple[float, float]]):
        plt.figure(figsize=(15, 15), dpi=200)

        nx.draw(
            self.G,
            pos_embeds,
            node_color="#84befe",
            edge_color="#818181",
            node_size=25,
            width=0.5,
            arrowsize=5,
            font_size=5,
            font_family="IosevkaTerm NF",
            **self.draw_args,
        )

        if self.save_figure is not None:
            plt.savefig(str(self.save_figure))
        else:
            plt.show()

    def draw_embedded_rerun(self, pos_embeds: dict[str, tuple[float, float]]):
        numeric_mapping = pd.DataFrame(
            {
                "labels": (unique_node_labels := np.unique(list(pos_embeds.keys()))),
                "indices": np.arange(len(unique_node_labels)),
            }
        ).set_index("labels")

        embedding_matrix = np.row_stack(
            [pos_embeds[node_label] for node_label in unique_node_labels]
        )  # |V| x 2(pos_x, pos_y)

        # Normalise (point sizes in Rerun get weird because of floating-point precision issues?)
        embedding_matrix -= embedding_matrix.mean(axis=0, keepdims=True)
        embedding_matrix /= np.amax(embedding_matrix, axis=0, keepdims=True)
        if self.rotate:
            embedding_matrix = embedding_matrix[:, [1, 0]]

        edge_matrix = np.row_stack(
            [
                [
                    numeric_mapping.loc[u_node].indices,
                    numeric_mapping.loc[v_node].indices,
                ]
                for u_node, v_node in self.G.edges()
            ]
        )  # |E| x 2(idx_u, idx_v)

        segments = embedding_matrix[edge_matrix.flatten(), :].reshape(
            len(edge_matrix), 2, 2
        )

        if colour_by_module := True:
            colors = [highlight(self.G.nodes[u]["module"]) for u in unique_node_labels]
        else:
            colors = [255, 255, 255, 255]

        rr.log(
            "callgraph/nodes",
            rr.Points2D(
                embedding_matrix,
                labels=unique_node_labels,
                colors=colors,
                draw_order=1,
                show_labels=True,
            ),
        )

        rr.log(
            "callgraph/edges",
            rr.LineStrips2D(
                list(segments),
                radii=rr.Radius.ui_points(0.5),
                draw_order=100,
            ),
        )

        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Spatial2DView(background=[9, 12, 8, 255]),
                auto_layout=True,
            )
        )

    @contextmanager
    def enter_session(self, xi: ExecInfo):
        with super().enter_session(xi):
            try:
                yield
            finally:
                # Embed into 2D space

                # self.G = self.G.subgraph(
                #     max(nx.weakly_connected_components(self.G), key=len)
                # )

                positions = [
                    graphviz_layout(self.G, prog="dot"),  # tree
                    # nx.spring_layout(self.G),
                    # nx.bfs_layout(
                    #     self.G,
                    #     "maaaaaaain",
                    # ),
                    # nx.kamada_kawai_layout(self.G),
                    # nx.spring_layout(self.G),
                ]

                for pos in positions:
                    pos = {node: (-xy[1], xy[0]) for node, xy in pos.items()}

                    match self.draw_backend:
                        case "networkx":
                            self.draw_embedded_mpl(pos)

                        case "rerun":
                            self.draw_embedded_rerun(pos)


class Export(CaptureCallgraph):
    def __init__(self, format: str):
        super().__init__()
        self.format = format

    @staticmethod
    @click.command()
    @click.option("--dot", "format", flag_value="dot", default=True)
    @click.option("--mermaid", "format", flag_value="mermaid")
    def cli(format: str):
        return Export(format=format)

    @contextmanager
    def enter_session(self, xi: ExecInfo):
        with super().enter_session(xi), self.hide_display():
            try:
                yield
            finally:
                match self.format:
                    case "dot":
                        print(self.networkx_to_dot(self.graph()))
                    case "mermaid":
                        mermaid_str = self.networkx_to_mermaid(self.G)
                        print(mermaid_str)
                    case _:
                        raise ValueError(f"{self.format}: unhandled graph format")

    @staticmethod
    def networkx_to_mermaid(G):
        mermaid = ["graph TD"]
        for edge in G.edges():
            src, dst = edge
            src_label = G.nodes[src].get("label", src)
            dst_label = G.nodes[dst].get("label", dst)
            mermaid.append(f"    {src}[{src_label}] --> {dst}[{dst_label}]")
        return "\n".join(mermaid)

    @staticmethod
    def networkx_to_dot(G):
        # buf = StringIO()
        # write_dot(G, buf)
        # return buf.getvalue()
        write_dot(G, "/tmp/f.txt")
