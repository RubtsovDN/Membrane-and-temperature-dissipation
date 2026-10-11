"""Computational DAGs for the unforced membrane and heat equations."""
import networkx as nx
import numpy as np
import sympy as sp


def equation_graph(kind):
    """Edges go from an input to an operation; args preserve argument order."""
    if kind == 'membrane':
        rows = [
            ('u', 'u', 'input', ()),
            ('c', 'c', 'input', ()),
            ('gamma', 'γ', 'input', ()),
            ('two', '2', 'constant', ()),
            ('lap', 'Δu', 'laplacian', ('u',)),
            ('c2', 'c²', 'square', ('c',)),
            ('ut', '∂ₜu', 'dt', ('u',)),
            ('elastic', 'c²Δu', 'multiply', ('c2', 'lap')),
            ('loss', '2γ∂ₜu', 'multiply', ('two', 'gamma', 'ut')),
            ('rhs', 'F = ∂ₜₜu', 'subtract', ('elastic', 'loss')),
        ]
        parameters = ['c', 'gamma']
    elif kind == 'heat':
        rows = [
            ('theta', 'θ', 'input', ()),
            ('alpha', 'α', 'input', ()),
            ('beta', 'β', 'input', ()),
            ('lap', 'Δθ', 'laplacian', ('theta',)),
            ('diffusion', 'αΔθ', 'multiply', ('alpha', 'lap')),
            ('loss', 'βθ', 'multiply', ('beta', 'theta')),
            ('rhs', 'F = ∂ₜθ', 'subtract', ('diffusion', 'loss')),
        ]
        parameters = ['alpha', 'beta']
    else:
        raise ValueError("kind must be 'membrane' or 'heat'")
    graph = nx.DiGraph(kind=kind, output='rhs', parameters=parameters)
    for name, label, op, args in rows:
        graph.add_node(name, label=label, op=op, args=list(args),
                       shape='scalar' if name in parameters + ['two', 'c2']
                       else 'scalar_field')
        for arg in args:
            graph.add_edge(arg, name)
    if kind == 'membrane':
        graph.nodes['two']['value'] = 2
    return graph


def graph_data(graph):
    """Return ordered node dictionary and Z[operation, input]."""
    order = list(nx.topological_sort(graph))
    A = nx.to_numpy_array(graph, nodelist=order, dtype=int, weight=None)
    payload = dict(graph.graph)
    payload.update(order=order, convention='Z[i,j]=1 iff j is input of i',
                   nodes={n: dict(graph.nodes[n]) for n in order},
                   Z=A.T.tolist())
    return payload, np.asarray(payload['Z'], dtype=int)


def evaluate_graph(graph, inputs, coords, time):
    """Evaluate with SymPy. coords contains spatial variables only."""
    values = {}
    for node in nx.topological_sort(graph):
        item = graph.nodes[node]
        op = item['op']
        args = [values[a] for a in item['args']]
        if op == 'input':
            value = sp.sympify(inputs[node])
        elif op == 'constant':
            value = sp.sympify(item['value'])
        elif op == 'laplacian':
            value = sum(sp.diff(args[0], x, 2) for x in coords)
        elif op == 'dt':
            value = sp.diff(args[0], time)
        elif op == 'square':
            value = args[0] ** 2
        elif op == 'multiply':
            value = sp.Mul(*args)
        elif op == 'subtract':
            value = args[0] - args[1]
        else:
            raise ValueError(f'Unknown operation: {op}')
        values[node] = value
    return values[graph.graph['output']]


def graph_positions(graph):
    """Deterministic layout: inputs at the bottom, result at the top."""
    positions = {}
    for level, generation in enumerate(nx.topological_generations(graph)):
        for j, node in enumerate(generation):
            positions[node] = (j - (len(generation) - 1) / 2, level)
    return positions
