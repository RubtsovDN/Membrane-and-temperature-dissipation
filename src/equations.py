# equations.py
"""Символьная постановка и DAG правой части уравнения."""

import networkx as nx
import sympy as sp


def symbolic_equations():
    x, y, t = sp.symbols("x y t", real=True)
    alpha, c = sp.symbols("alpha c", positive=True)
    beta, gamma = sp.symbols("beta gamma", nonnegative=True)

    temperature = sp.Function("T")(x, y, t)
    displacement = sp.Function("u")(x, y, t)

    def laplace(field):
        return sp.diff(field, x, 2) + sp.diff(field, y, 2)

    return {
        "heat_base": sp.Eq(
            sp.diff(temperature, t),
            alpha * laplace(temperature),
        ),
        "heat_modified": sp.Eq(
            sp.diff(temperature, t),
            alpha * laplace(temperature) - beta * temperature,
        ),
        "wave_base": sp.Eq(
            sp.diff(displacement, t, 2),
            c**2 * laplace(displacement),
        ),
        "wave_modified": sp.Eq(
            sp.diff(displacement, t, 2),
            c**2 * laplace(displacement)
            - gamma * sp.diff(displacement, t),
        ),
    }


def equation_graph(kind):
    if kind not in {"heat", "wave"}:
        raise ValueError("kind должен быть heat или wave.")

    graph = nx.DiGraph()
    field = "T" if kind == "heat" else "u"

    labels = {
        "field": field,
        "dxx": f"d²{field}/dx²",
        "dyy": f"d²{field}/dy²",
        "laplace": "sum: Laplacian",
        "d": "alpha" if kind == "heat" else "c²",
        "spatial": "d * Laplacian",
        "loss": "beta" if kind == "heat" else "gamma",
        "damping": "loss * state",
        "rhs": "spatial - damping",
        "rate": "T_t" if kind == "heat" else "v_t",
    }

    if kind == "wave":
        labels.update({"velocity": "v", "u_rate": "u_t"})

    for node, label in labels.items():
        graph.add_node(node, label=label)

    graph.add_edges_from([
        ("field", "dxx"),
        ("field", "dyy"),
        ("dxx", "laplace"),
        ("dyy", "laplace"),
        ("laplace", "spatial"),
        ("d", "spatial"),
        ("loss", "damping"),
        ("spatial", "rhs"),
        ("damping", "rhs"),
        ("rhs", "rate"),
    ])

    if kind == "heat":
        graph.add_edge("field", "damping")
    else:
        graph.add_edges_from([
            ("velocity", "damping"),
            ("velocity", "u_rate"),
        ])

    return graph


def graph_representations(kind):
    graph = equation_graph(kind)
    nodes = list(nx.topological_sort(graph))

    return {
        "graph": graph,
        "nodes": nodes,
        "adjacency": nx.to_numpy_array(
            graph, nodelist=nodes, dtype=int
        ),
        "dictionary": {
            node: {
                "label": graph.nodes[node]["label"],
                "children": list(graph.successors(node)),
            }
            for node in nodes
        },
    }