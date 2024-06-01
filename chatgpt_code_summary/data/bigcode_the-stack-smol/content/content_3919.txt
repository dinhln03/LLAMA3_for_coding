"""
Visualizing bidirectionnal Dijkstra
using matplotlib
"""
import sys
from dataclasses import dataclass
from heapq import heappush, heappop
from itertools import permutations
from collections import defaultdict
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from dijkstra import (
    Node,
    generate_random_graph,
    build_shortest_path,
    dijkstra,
)


@dataclass
class Context:
    distances: dict
    previous: dict
    node: None
    visited_nodes: set


def dijkstra_iterator(nodes: list[Node], src_id: int, hf=lambda x: 0.0):
    """
    Internal loop of the Dijkstra algorithm
    as a step by step iterator
    hf is an optional heuristic
    """
    visited_nodes = set()
    h: list[tuple[float, Node]] = []
    previous = dict()
    distances = defaultdict(lambda: sys.maxsize)
    distances[src_id] = hf(nodes[src_id])
    ctx: Context = Context(
        previous=previous,
        distances=distances,
        node=None,
        visited_nodes=visited_nodes,
    )

    heappush(h, (0.0, nodes[src_id]))

    while h:
        _, node = heappop(h)

        if node.id in visited_nodes:
            continue

        dist = distances[node.id]

        for n, d in (
            (nodes[k], v)
            for k, v in node.neighbours.items()
            if k not in visited_nodes
        ):
            new_dist = dist + d
            cost = new_dist + hf(n) - hf(node)
            if cost <= distances[n.id]:
                distances[n.id] = cost
                previous[n.id] = node.id

                heappush(h, (cost, n))

        visited_nodes.add(node.id)
        ctx.node = node
        yield ctx

    ctx.node = None
    yield ctx


def dijkstra_forward(
    nodes: list[Node], src_id: int, dst_id: int, hf=lambda x: 0.0
) -> list[int]:
    """
    'classical' forward Dijkstra but based on our iterator.
    """
    coro = dijkstra_iterator(nodes, src_id, hf=hf)
    for ctx in coro:
        if ctx.node is None:
            return [], []
        elif ctx.node.id == dst_id:
            return ctx.distances[dst_id], list(
                build_shortest_path(ctx.previous, dst_id, src_id)
            )


def bidir_dijkstra(
    nodes: list[Node],
    src_id: int,
    dst_id: int,
    hff=lambda _: 0.0,
    hfb=lambda _: 0.0,
    consistent: bool = True,
) -> list[int]:
    """
    bidirectionnal dijkstra, we search from both start => end
    and end => start using two iterators.
    hff and hfb are optional heuristics
    for respectively the forward and backward iterators
    (for later bidir A*)
    """
    forward = dijkstra_iterator(nodes, src_id, hf=hff)
    backward = dijkstra_iterator(nodes, dst_id, hf=hfb)

    shortest = sys.maxsize
    forward_node = backward_node = None
    f = []
    b = []
    for idx, (ctx_forward, ctx_backward) in enumerate(zip(forward, backward)):

        if any(x.node is None for x in (ctx_forward, ctx_backward)):
            # no path between the two nodes
            return [], [], (f, b)

        f.append(ctx_forward.node)
        b.append(ctx_backward.node)

        if forward_node and (
            not consistent
            or sum(
                x.distances[x.node.id] - hf(x.node)
                for x, hf in ((ctx_forward, hff), (ctx_backward, hfb))
            )
            >= shortest
        ):

            forward_path = build_shortest_path(
                ctx_forward.previous, forward_node.id, src_id
            )
            backward_path = build_shortest_path(
                ctx_backward.previous, backward_node.id, dst_id
            )[::-1]
            path = forward_path + backward_path
            return (
                shortest,
                path,
                (f, b),
            )

        else:
            for (ctx, hf), (ctx2, hf2) in permutations(
                ((ctx_forward, hff), (ctx_backward, hfb)), 2
            ):
                for n, d in ctx.node.neighbours.items():
                    if n in ctx2.visited_nodes:
                        distance = (
                            ctx.distances[ctx.node.id]
                            + ctx2.distances[n]
                            + d
                            - hf(ctx.node)
                            - hf2(nodes[n])
                        )

                        if distance < shortest:
                            shortest = distance
                            forward_node = (
                                ctx.node if ctx is ctx_forward else nodes[n]
                            )
                            backward_node = (
                                ctx.node if ctx is ctx_backward else nodes[n]
                            )
                            print(
                                f'Iter_{idx}: contact between {forward_node}->{backward_node} with d={shortest}'
                            )


class Animator:
    """
    Builds an animation from
    a bidir shortest path finder.
    """

    def __init__(self, nodes: list[Node], title='', draw_edges=True) -> None:
        self.fig, self.ax = plt.subplots()
        plt.title(title)
        plt.tight_layout()
        self.ax.set_aspect('equal')
        self.i = True
        if draw_edges:
            edges = {
                tuple(sorted((n.id, x))) for n in nodes for x in n.neighbours
            }
            for edge in edges:
                from_node, to_node = [nodes[x] for x in edge]
                x = [n.x for n in (from_node, to_node)]
                y = [n.y for n in (from_node, to_node)]
                plt.plot(x, y, color='gray', linewidth=0.5)

        x, y = [n.x for n in nodes], [n.y for n in nodes]
        self.ax.scatter = plt.scatter(
            x,
            y,
            c=[0 for _ in range(len(x))],
            s=[30] + [10] * (len(nodes) - 2) + [30],
            vmin=0,
            vmax=3,
            cmap=matplotlib.colors.ListedColormap(
                ['grey', 'springgreen', 'red', 'white']
            ),
        )
        self._colors = self.ax.scatter.get_array()

        for n in nodes:
            if not n.neighbours:
                self._colors[n.id] = 3

    def update(self, nodes: tuple[Node, Node, list[Node]]):
        """
        Updates the plot with a tuple of nodes (forward, backward, shortest_path)
        """
        f, b, s = nodes

        if not s:
            self._colors[f.id] = 1
            self._colors[b.id] = 2
            self.ax.scatter.set_array(self._colors)
            return (self.ax.scatter,)
        else:
            x = [n.x for n in s]
            y = [n.y for n in s]

            if self.i:
                c = 'green'
            else:
                c = 'orange'

            ap = self.ax.plot(x, y, color=c, linewidth=2)
            self.i = not (self.i)
            return ap


def make_animated_gif(
    title: str,
    g: list[Node],
    dst_file: str,
    fs: list[Node],
    bs: list[Node],
    shortest: list[Node],
    draw_edges: bool = True,
    writer: str = 'ffmpeg',
    interval: int = 250,
    blinking_ratio=0.5,
):
    """
    Makes an animated gif out of two sequences of forward (fs) and backward (bs)
    path-finding algorithm. The final shortest path will be blinked.
    """
    anim = Animator(g, title=title, draw_edges=draw_edges)

    def node_gen():
        for fn, bn in zip(fs, bs):
            yield fn, bn, []

        res = [g[i] for i in shortest]
        for _ in range(int(len(fs) * blinking_ratio)):
            yield _, _, res

    ani = animation.FuncAnimation(
        anim.fig,
        anim.update,
        node_gen(),
        interval=interval,
        blit=True,
        repeat_delay=500,
        save_count=len(fs) * 2,
    )
    ani.save(f'imgs/{dst_file}', writer=writer)


if __name__ == '__main__':
    # sanity check on the iterator versus 'simple' implementation
    g = generate_random_graph(100, connect_probability=0.1)
    cost, sp = dijkstra_forward(g, 0, len(g) - 1)
    cost2, sp2 = dijkstra(g, 0, len(g) - 1)
    # we also compare our bidir version agaisnt the other two ^^
    cost3, sp3, (f, b) = bidir_dijkstra(g, 0, len(g) - 1)

    # and against a backward run only
    cost4, sp4 = dijkstra_forward(g, len(g) - 1, 0)
    sp4 = sp4[::-1]

    print(cost, cost2, cost3, cost4)

    for p in (sp, sp2, sp4, sp3):
        print(' -> '.join(str(p) for p in p))

    assert sp == sp2 == sp3 == sp4

    make_animated_gif(
        f'Bidir Dijkstra n={len(f)}', g, 'bidir_100.gif', f, b, sp3
    )
