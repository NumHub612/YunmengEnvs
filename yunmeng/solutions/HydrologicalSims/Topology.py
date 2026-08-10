# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Unified management and traversal of tree topology inside the watershed。

Responsibilities:
* Adjacency relationships coming from upstream/downstream edges;
* Tree validity checks (no cycles, at most one downstream, fully connected);
* DFS/BFS traversals along the topology;
* Queries such as topological order (leaf → root, for stepwise progression),
  path to root, upstream closure, etc.
"""

from __future__ import annotations
from collections import deque
from typing import Callable, Iterator


class HydroTopology:
    """A validated rooted tree (arborescence) of node names."""

    def __init__(self, node_names: list[str], edges: list[tuple[str, str]]):
        """
        Args:
            node_names: all node names (unique).
            edges: (upstream, downstream) pairs.
        """
        if len(set(node_names)) != len(node_names):
            dup = [n for n in set(node_names) if node_names.count(n) > 1]
            raise ValueError(f"Duplicate node names: {dup}.")
        if not node_names:
            raise ValueError("No nodes configured.")

        self._nodes = list(node_names)
        self._up: dict[str, list[str]] = {n: [] for n in node_names}
        self._down: dict[str, str] = {n: None for n in node_names}

        for u, v in edges:
            for n in (u, v):
                if n not in self._up:
                    raise ValueError(f"Edge references unknown node '{n}'.")
            if u == v:
                raise ValueError(f"Self-loop on node '{u}'.")
            if self._down[u] is not None:
                raise ValueError(
                    f"Node '{u}' has more than one downstream "
                    f"('{self._down[u]}' and '{v}') — not a tree."
                )
            self._down[u] = v
            self._up[v].append(u)

        # cycle check first (a pure cycle has no root at all)
        for n in node_names:
            seen = set()
            cur = n
            while cur is not None:
                if cur in seen:
                    raise ValueError(f"Cycle detected along node '{n}'.")
                seen.add(cur)
                cur = self._down[cur]

        roots = [n for n in node_names if self._down[n] is None]
        if len(roots) != 1:
            raise ValueError(
                f"A tree topology needs exactly one root (outlet); "
                f"found {len(roots)}: {roots}."
            )
        self._root = roots[0]

        # connectivity: every node must reach the root
        for n in node_names:
            if self._root not in self._walk(n):
                raise ValueError(
                    f"Node '{n}' is not connected to the root '{self._root}'."
                )

        # post-order (leaves -> root) for step-wise propagation
        self._topo_order = self._post_order(self._root)

    def _walk(self, node: str) -> set:
        seen = set()
        cur = node
        while cur is not None:
            seen.add(cur)
            cur = self._down[cur]
        return seen

    def _post_order(self, root: str) -> list[str]:
        order, seen = [], set()
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if node in seen:
                continue
            seen.add(node)
            stack.append((node, True))
            for u in self._up[node]:
                stack.append((u, False))
        return order

    # -- basic queries ------------------------------

    @property
    def nodes(self) -> list[str]:
        return list(self._nodes)

    @property
    def root(self) -> str:
        return self._root

    @property
    def topo_order(self) -> list[str]:
        """Leaves -> root order for step-wise propagation."""
        return list(self._topo_order)

    def upstreams(self, node: str) -> list[str]:
        return list(self._up[node])

    def downstream(self, node: str) -> str:
        return self._down[node]

    def is_leaf(self, node: str) -> bool:
        return not self._up[node]

    def leaves(self) -> list[str]:
        return [n for n in self._nodes if self.is_leaf(n)]

    # -- traversals ---------------------------------

    def dfs(
        self,
        start: str = None,
        direction: str = "up",
        visit: Callable[[str, int], None] = None,
    ) -> Iterator[str]:
        """Depth-first traversal along the topology.

        Args:
            start: starting node (default: root).
            direction: "up" — towards headwaters (follow upstreams);
                       "down" — towards the outlet (follow downstream).
            visit: optional callback (node, depth).
        Yields:
            node names in DFS pre-order.
        """
        self._check_direction(direction)
        start = start or self._root
        self._check_node(start)

        stack = [(start, 0)]
        seen = set()
        while stack:
            node, depth = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if visit:
                visit(node, depth)
            yield node
            neighbours = self._neighbours(node, direction)
            for nb in reversed(neighbours):
                if nb not in seen:
                    stack.append((nb, depth + 1))

    def bfs(
        self,
        start: str = None,
        direction: str = "up",
        visit: Callable[[str, int], None] = None,
    ) -> Iterator[str]:
        """Breadth-first traversal; signature mirrors :meth:`dfs`."""
        self._check_direction(direction)
        start = start or self._root
        self._check_node(start)

        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            node, depth = queue.popleft()
            if visit:
                visit(node, depth)
            yield node
            for nb in self._neighbours(node, direction):
                if nb not in seen:
                    seen.add(nb)
                    queue.append((nb, depth + 1))

    # -- derived queries ----------------------------

    def path_to_root(self, node: str) -> list[str]:
        """[node, ..., root] following downstream links."""
        self._check_node(node)
        path = [node]
        while self._down[path[-1]] is not None:
            path.append(self._down[path[-1]])
        return path

    def upstream_closure(self, node: str) -> list[str]:
        """All nodes draining into *node* (inclusive), DFS order."""
        return list(self.dfs(node, direction="up"))

    def contributing_leaves(self, node: str) -> list[str]:
        """Leaf (headwater) nodes upstream of *node*."""
        return [n for n in self.upstream_closure(node) if self.is_leaf(n)]

    def depth(self, node: str) -> int:
        """Number of reaches from *node* down to the root."""
        return len(self.path_to_root(node)) - 1

    # -- helpers ------------------------------------

    def _neighbours(self, node: str, direction: str) -> list[str]:
        if direction == "up":
            return self._up[node]
        nxt = self._down[node]
        return [nxt] if nxt is not None else []

    def _check_node(self, node: str):
        if node not in self._up:
            raise KeyError(f"Unknown node '{node}'.")

    @staticmethod
    def _check_direction(direction: str):
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got '{direction}'.")

    def describe(self) -> str:
        """Indented tree rendering (root first)."""
        lines = []

        def render(node: str, depth: int):
            lines.append("  " * depth + node)
            for u in self._up[node]:
                render(u, depth + 1)

        render(self._root, 0)
        return "\n".join(lines)
