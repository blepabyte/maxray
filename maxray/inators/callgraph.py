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
import numpy as np

from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Literal

import rerun as rr
import rerun.blueprint as rrb

import click


class CaptureCallgraph(BaseInator):
    def __init__(self, all_calls=False):
        super().__init__()
        self.G = nx.DiGraph()
        self.all_calls = all_calls

    def xray(self, x, ray: Ray):
        src_id = ray.ctx.fn_context.name

        # raise Break()
        # TODO: handle noncompiled and errored functions

        # TODO: identify functions properly/qualname/modules
        match ray.called():
            case {"target": _} if isinstance(x, type):
                # __init__
                dst_id = f"{x.__name__}()"
                self.G.add_edge(src_id, dst_id)

            case {"fn_compile_id": compile_id}:
                dst_id = FunctionStore.get(compile_id).data.qualname
                self.G.add_edge(src_id, dst_id)

            case {"target": _} if self.all_calls:
                try:
                    dst_id = x.__getattribute__("__name__")
                    self.G.add_edge(src_id, dst_id)

                except Exception:
                    pass


class Draw(CaptureCallgraph):
    def __init__(
        self,
        backend: Literal["networkx", "rerun"],
        draw_args: dict,
        save_figure: Optional[Path],
    ):
        super().__init__()
        self.draw_backend = backend
        if self.draw_backend == "rerun":
            rr.init("callgraph", spawn=True)

        self.draw_args = draw_args
        self.save_figure = save_figure

    @staticmethod
    @click.command()
    @click.option("--labels", is_flag=True)
    @click.option("--save", type=Path)
    @click.option("--nx", "backend", flag_value="networkx", default=True)
    @click.option("--rr", "--rerun", "backend", flag_value="rerun")
    def cli(labels: bool, backend: Literal["networkx", "rerun"], save: Optional[Path]):
        draw_args = {}
        if labels:
            draw_args["with_labels"] = True

        # TODO: post-processing view for just the __init__ dependency graph?
        return Draw(draw_args=draw_args, backend=backend, save_figure=save)

    @contextmanager
    def enter_session(self, xi: ExecInfo):
        with super().enter_session(xi):
            try:
                yield
            finally:
                # Embed into 2D space
                pos = graphviz_layout(self.G, prog="dot")  # tree
                pos = {node: (-xy[1], xy[0]) for node, xy in pos.items()}

                match self.draw_backend:
                    case "networkx":
                        plt.figure(figsize=(15, 15), dpi=200)
                        # TODO: configure layout when graphviz not available
                        # pos = graphviz_layout(self.G) # sprawl
                        # pos = nx.spring_layout(self.G)
                        # pos = nx.kamada_kawai_layout(self.G)
                        # pos = nx.bfs_layout(
                        #     self.G.subgraph(nx.node_connected_component(self.G, "maaaaaaain")),
                        #     "maaaaaaain",
                        # )

                        nx.draw(
                            self.G,
                            pos,
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
                    case "rerun":
                        labels = []
                        positions = []
                        for label, (x, y) in pos.items():
                            labels.append(label)
                            positions.append([y / 10, x])

                        segments = []
                        for u, v in self.G.edges():
                            segments.append(
                                np.array(
                                    [
                                        [pos[u][1], pos[u][0]],
                                        [pos[v][1], pos[v][0]],
                                    ]
                                )
                            )

                        rr.log(
                            "nodes",
                            rr.Points2D(np.array(positions) / 1000, labels=labels),
                        )
                        rr.log("edges", rr.LineStrips2D(segments))


class Export(CaptureCallgraph):
    def __init__(self):
        super().__init__()

    @staticmethod
    @click.command()
    @click.option("--flag", is_flag=True)
    @click.option("--save", type=Path)
    def cli():
        return Export()

    @contextmanager
    def enter_session(self, xi: ExecInfo):
        with super().enter_session(xi), S.display.hidden():
            try:
                yield
            finally:
                mermaid_str = self.networkx_to_mermaid(self.G)
                print(mermaid_str)

    @staticmethod
    def networkx_to_mermaid(G):
        mermaid = ["graph TD"]
        for edge in G.edges():
            src, dst = edge
            src_label = G.nodes[src].get("label", src)
            dst_label = G.nodes[dst].get("label", dst)
            mermaid.append(f"    {src}[{src_label}] --> {dst}[{dst_label}]")
        return "\n".join(mermaid)
